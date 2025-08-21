"""
Forked from apple's ademamix jax implementation:
https://github.com/apple/ml-ademamix
for the purposes of submitting to the algoperf benchmark.
| Adapted from optax's implementation of AdamW:
| https://github.com/google-deepmind/optax/blob/b75644809f2f68fc11f42d4395a5753e11e92e80/optax/_src/alias.py#L548#L675
"""
import functools
from typing import (
        Any,
        Callable,
        Dict,
        Iterator,
        List,
        NamedTuple,
        Optional,
        Tuple,
        Union,
        )

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
import optax
from flax import jax_utils
from jax import lax

from optax._src import transform, combine, base, numerics, utils
from optax import tree_utils as otu

from algoperf import spec


HPARAMS = {
        'alpha': 8.0,
        'alpha_start': 0,
        'warmup': 10,
        'beta_end': 0.9999,
        'beta_start': 0.9,
        'learning_rate': 0.01,
        'b1': 0.9,
        'b2': 0.999,
        'b3': 0.9999,
        'eps': 1e-8,
        'eps_root': 0.0,
        'weight_decay': 0.01,
        }
        
_GRAD_CLIP_EPS = 1e-6

def alpha_scheduler(alpha, alpha_start=0, warmup=0):
    def schedule(step):
        is_warmup = (step < warmup).astype(jnp.float32)
        a = step / float(warmup)
        return is_warmup * ((1.0-a) * alpha_start + a * alpha) + alpha * (1.0-is_warmup)

    return schedule


def beta3_scheduler(beta_end, beta_start=0, warmup=0):

    def f(beta):
        return jnp.log(0.5)/jnp.log(beta)-1

    def f_inv(t):
        return jnp.power(0.5, 1/(t+1))

    def schedule(step):
        is_warmup = (step < warmup).astype(jnp.float32)
        alpha = step / float(warmup)
        return is_warmup * f_inv((1.0-alpha) * f(beta_start) + alpha * f(beta_end)) + beta_end * (1.0-is_warmup)

    return schedule


class ScaleByAdemamixState(NamedTuple):
  """State for the AdEMAMix algorithm."""
  count: chex.Array  
  count_m2: chex.Array  
  m1: base.Updates
  m2: base.Updates
  nu: base.Updates


def ademamix(lr, b1=0.9, b2=0.999, b3=0.9999, alpha=5.0, b3_scheduler=None, alpha_scheduler=None,
             eps=1e-8, eps_root=0.0, weight_decay=0.0, mu_dtype=None, mask=None):
  r"""AdEMAMix.

    Args:
        lr: A global scaling factor, either fixed or evolving along
            iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        b1: Exponential decay rate to track the fast EMA.
        b2: Exponential decay rate to track the second moment of past gradients.
        b3: Exponential decay rate to track the slow EMA.
        alpha: Mixing coeficient use for the linear combination of the fast and slow EMAs.
        b3_scheduler: an optional scheduler function, given a timestep, returns the 
            value of b3. Use `beta3_scheduler(b3,b1,T_b3)` to follow the AdEMAMix paper. 
        alpha_scheduler: an optional scheduler function, given a timestep, returns the 
            value of alpha. Use `alpha_scheduler(alpha,0,T_alpha)` to follow the 
            AdEMAMix paper. 
        eps: A small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            instance when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
            weight decay is multiplied with the learning rate. This is consistent
            with other frameworks such as PyTorch, but different from
            (Loshchilov et al, 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.

    Returns:
        The corresponding `GradientTransformation`.
  """
  return combine.chain(
    scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps, eps_root, mu_dtype),
    transform.add_decayed_weights(weight_decay, mask),
    transform.scale_by_learning_rate(lr),
  )


def scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps=1e-8, eps_root=0.0, mu_dtype=None):

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    m1 = tree_zeros_like(params, dtype=mu_dtype)   # fast EMA
    m2 = tree_zeros_like(params, dtype=mu_dtype)   # slow EMA
    nu = tree_zeros_like(params, dtype=mu_dtype)   # second moment estimate
    return ScaleByAdemamixState(count=jnp.zeros([], jnp.int32), count_m2=jnp.zeros([], jnp.int32), m1=m1, m2=m2, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
    c_alpha = alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
    m1 = tree_update_moment(updates, state.m1, b1, 1) # m1 = b1 * m1 + (1-b1) * updates
    m2 = tree_update_moment(updates, state.m2, c_b3, 1)
    nu = tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    count_m2_inc = numerics.safe_int32_increment(state.count_m2)
    m1_hat = tree_bias_correction(m1, b1, count_inc)
    nu_hat = tree_bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(lambda m1_, m2_, v_: (m1_+c_alpha*m2_)/(jnp.sqrt(v_+eps_root)+eps), m1_hat, m2, nu_hat)
    mu1 = tree_cast(m1, mu_dtype)
    mu2 = tree_cast(m2, mu_dtype)
    return updates, ScaleByAdemamixState(count=count_inc, count_m2=count_m2_inc, m1=m1, m2=m2, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def tree_cast(tree, dtype):
  """Cast tree to given dtype, skip if None."""
  if dtype is not None:
    return jtu.tree_map(lambda t: t.astype(dtype), tree)
  else:
    return tree


def tree_zeros_like(
    tree,
    dtype = None,
):
  """Creates an all-zeros tree with the same structure.

  Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of zeros.

  Returns:
    an all-zeros tree with the same structure as ``tree``.
  """
  return jtu.tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jtu.tree_map(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return numerics.abs_sq(g) ** half_order

  return jtu.tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


@functools.partial(jax.jit, inline=True)
def tree_bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)

@functools.partial(
        jax.pmap,
        axis_name='batch',
        in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        static_broadcasted_argnums=(0, 1),
        )
def pmapped_train_step(
        workload,
        opt_update_fn,
        model_state,
        optimizer_state,
        current_param_container,
        batch,
        rng,
        grad_clip,
        label_smoothing
        ):
    def _loss_fn(params):
        logits, new_model_state = workload.model_fn(
                params,
                batch,
                model_state,
                spec.ForwardPassMode.TRAIN,
                rng,
                update_batch_norm=True,
                )
        loss_dict = workload.loss_fn(
                label_batch=batch['targets'],
                logits_batch=logits,
                mask_batch=batch.get('weights'),
                label_smoothing=label_smoothing,
                )
        summed_loss = loss_dict['summed']
        n_valid_examples = loss_dict['n_valid_examples']
        return summed_loss, (n_valid_examples, new_model_state)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
            current_param_container
            )
    (summed_loss, n_valid_examples, grad) = lax.psum(
            (summed_loss, n_valid_examples, grad), axis_name='batch'
            )
    loss = summed_loss / n_valid_examples
    grad = jax.tree.map(lambda x: x / n_valid_examples, grad)
    grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grad))
            )
    if grad_clip is not None:
        grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
        grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
        grad = jax.tree.map(lambda x: x * grad_scaling_factor, grad)

    updates, new_optimizer_state = opt_update_fn(
            grad, optimizer_state, current_param_container
            )
    updated_params = optax.apply_updates(current_param_container, updates)
    return new_optimizer_state, updated_params, new_model_state, loss, grad_norm

def update_params(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        batch: Dict[str, spec.Tensor],
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        train_state: Optional[Dict[str, Any]] = None,
        ) -> spec.UpdateReturn:
    del current_params_types
    del loss_type
    del train_state
    del eval_results
    del hyperparameters

    hyperparameters = HPARAMS

    optimizer_state, opt_update_fn = optimizer_state
    per_device_rngs = jax.random.split(rng, jax.local_device_count())
    if hasattr(hyperparameters, 'label_smoothing'):
        label_smoothing = hyperparameters['label_smoothing']
    else:
        label_smoothing = 0.0
    if hasattr(hyperparameters, 'grad_clip'):
        grad_clip = hyperparameters['grad_clip']
    else:
        grad_clip = None
        outputs = pmapped_train_step(
                workload,
                opt_update_fn,
                model_state,
                optimizer_state,
                current_param_container,
                batch,
                per_device_rngs,
                grad_clip,
                label_smoothing,
                )
        new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

        if global_step % 100 == 0 and workload.metrics_logger is not None:
            workload.metrics_logger.append_scalar_metrics(
                    {
                        'loss': loss[0],
                        'grad_norm': grad_norm[0],
                        },
                    global_step
                    )
        return (new_optimizer_state, opt_update_fn), new_params, new_model_state

def prepare_for_eval(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        ) -> spec.UpdateReturn:
    del workload
    del hyperparameters
    del current_params_types
    del loss_type
    del eval_results
    del global_step
    del rng
    return (optimizer_state, current_param_container, model_state)

def get_batch_size(workload_name):
    if workload_name == 'criteo1tb':
        return 262_144
    elif workload_name == 'fastmri':
        return 32
    elif workload_name == 'imagenet_resnet':
        return 1024
    elif workload_name == 'imagenet_resnet_silu':
        return 512
    elif workload_name == 'imagenet_resnet_gelu':
        return 512
    elif workload_name == 'imagenet_vit':
        return 1024
    elif workload_name == 'librispeech_conformer':
        return 256
    elif workload_name == 'librispeech_deepspeech':
        return 256
    elif workload_name == 'ogbg':
        return 512
    elif workload_name == 'wmt':
        return 128
    elif workload_name == 'mnist':
        return 16
    elif workload_name == 'cifar':
        return 2048
    else:
        raise ValueError(f'Unsupported workload name: {workload_name}.')

def data_selection(
        workload: spec.Workload,
        input_queue: Iterator[Dict[str, spec.Tensor]],
        optimizer_state: spec.OptimizerState,
        current_param_container: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        global_step: int,
        rng: spec.RandomState,
        ) -> Dict[str, spec.Tensor]:
    del workload
    del optimizer_state
    del current_param_container
    del model_state
    del hyperparameters
    del global_step
    del rng
    batch = next(input_queue)
    return batch

def init_optimizer_state(
        workload: spec.Workload,
        model_params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        rng: spec.RandomState,
        ) -> spec.OptimizerState:
    del model_params
    del model_state
    del rng
    params_zeros_like = jax.tree.map(
            lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes
            )
    lr = HPARAMS['learning_rate']
    b1 = HPARAMS['b1']
    b2 = HPARAMS['b2']
    b3 = HPARAMS['b3']
    alpha = HPARAMS['alpha']
    warmup = HPARAMS['warmup']
    f_b3 = beta3_scheduler(b3, beta_start=b1, warmup=warmup)
    f_a = alpha_scheduler(alpha, alpha_start=0, warmup=warmup)
    weight_decay = HPARAMS['weight_decay']
    opt_init_fn, opt_update_fn = ademamix(lr=lr,
                                          b1=b1,
                                          b2=b2,
                                          b3=b3,
                                          alpha=alpha,
                                          b3_scheduler=f_b3,
                                          alpha_scheduler=f_a,
                                          weight_decay=weight_decay
                                          )
    optimizer_state = opt_init_fn(params_zeros_like)
    return jax_utils.replicate(optimizer_state), opt_update_fn

if __name__ == "__main__": # dummy test
   
    def f(x): return jnp.sum(x ** 2)  # simple quadratic function

    alpha = 8.0
    b1, b2, b3 = 0.9, 0.999, 0.9999

    f_a = alpha_scheduler(alpha, alpha_start=0, warmup=10)
    f_b3 = beta3_scheduler(b3, beta_start=b1, warmup=10)

    solver = ademamix(lr=0.01, 
                      b1=b1, 
                      b2=b2, 
                      b3=b3, 
                      alpha=alpha, 
                      b3_scheduler=f_b3, 
                      alpha_scheduler=f_a,
                      weight_decay=0.01)
    
    params = jnp.array([1., 2., 3.])
    print('Objective function: {:.2f}'.format(f(params)))
    opt_state = solver.init(params)
    for itr in range(100):
        grad = jax.grad(f)(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        if itr % 5 == 0:
           print('Objective function: {:.2f}'.format(f(params)))
    print(params)
