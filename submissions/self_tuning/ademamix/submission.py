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
from flax import traverse_util as tu
from jax import lax

from optax._src import transform, combine, base, numerics, utils
from optax import tree_utils as otu

from algoperf import spec, jax_sharding_utils


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
        'dropout_rate': 0.1,
        }
        
_GRAD_CLIP_EPS = 1e-6


def alpha_scheduler(alpha, alpha_start=0, warmup=0):
    warmup_fn = optax.linear_schedule(init_value=alpha_start, end_value=alpha, transition_steps=warmup)
    constant_fn = optax.constant_schedule(alpha)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[warmup])
    return schedule_fn


def beta3_scheduler(beta_end, beta_start=0, warmup=0):

    def f(beta):
        return jnp.log(0.5)/jnp.log(beta)-1

    def f_inv(t):
        return jnp.power(0.5, 1/(t+1))

    def warmup_fn(step):
        frac = 1 - step / warmup
        return f_inv( frac * f(beta_start) + (1 - frac) * f(beta_end))

    constant_fn = optax.constant_schedule(beta_end)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[warmup])
    return schedule_fn


class ScaleByAdemamixState(NamedTuple):
  """State for the AdEMAMix algorithm."""
  count: chex.Array  
  count_m2: chex.Array  
  m1: optax.Updates
  m2: optax.Updates
  nu: optax.Updates

def _path_matches_embedding(path_segments: tuple) -> bool:
    return any(("embedding_table" in s) or ("embedding" in s) for s in path_segments)

def build_embedding_name_mask(params_tree):
    flat = tu.flatten_dict(params_tree, keep_empty_nodes=True)
    mask_flat = {}
    for path, leaf in flat.items():
        mask_flat[path] = _path_matches_embedding(path)
    return tu.unflatten_dict(mask_flat)

def _choose_sharding(mask_tree, target_tree, sharded, replicated):
    return jax.tree.map(
            lambda m, _: sharded if m else replicated, mask_tree, target_tree)

def create_ademamix_sharding_from_names(
        optimizer_state: ScaleByAdemamixState,
        params_tree,
        replicated,
        sharded
        ) -> ScaleByAdemamixState:
    embed_mask = build_embedding_name_mask(params_tree)
    m1_sharding = _choose_sharding(embed_mask, optimizer_state.m1, sharded, replicated)
    m2_sharding = _choose_sharding(embed_mask, optimizer_state.m2, sharded, replicated)
    nu_sharding = _choose_sharding(embed_mask, optimizer_state.nu, sharded, replicated)
    return ScaleByAdemamixState(
            count=replicated,
            count_m2=replicated,
            m1=m1_sharding,
            m2=m2_sharding,
            nu=nu_sharding
            )


def ademamix(lr, b1=0.9, b2=0.999, b3=0.9999, alpha=5.0, b3_scheduler=None, alpha_scheduler=None,
             eps=1e-8, eps_root=0.0, weight_decay=0.0, mask=None):
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
    scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps, eps_root),
    transform.add_decayed_weights(weight_decay, mask),
    transform.scale_by_learning_rate(lr),
  )


def scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps=1e-8, eps_root=0.0):

  def init_fn(params):
    m1 = jax.tree.map(jnp.zeros_like, params)   # fast EMA
    m2 = jax.tree.map(jnp.zeros_like, params)   # slow EMA
    nu = jax.tree.map(jnp.zeros_like, params)   # second moment estimate
    return ScaleByAdemamixState(count=jnp.zeros([], jnp.int32), count_m2=jnp.zeros([], jnp.int32), m1=m1, m2=m2, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3
    c_alpha = alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha
    m1 = _update_moment(updates, state.m1, b1, 1) # m1 = b1 * m1 + (1-b1) * updates
    m2 = _update_moment(updates, state.m2, c_b3, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    # count_inc = numerics.safe_int32_increment(state.count)
    count_m2 = state.count_m2 + jnp.array(1, dtype=jnp.int32)
    # count_m2_inc = numerics.safe_int32_increment(state.count_m2)
    m1_hat = _bias_correction(m1, b1, count)
    nu_hat = _bias_correction(nu, b2, count)
    updates = jax.tree.map(lambda m1_, m2_, v_: (m1_+c_alpha*m2_)/(jnp.sqrt(v_+eps_root)+eps), m1_hat, m2, nu_hat)
    return updates, ScaleByAdemamixState(count=count, count_m2=count_m2, m1=m1, m2=m2, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree.map(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)



def _bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree.map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)

  def create_optimizer_sharding(optimizer_state, replicated, sharded):
    """
    Create sharding spec for optimizer

    Args:
        optimizer_state: The optimizer state structure
        replicated: Sharding spec for replicated data
        sharded: Sharding spec for batch sharded data

    Returns:
        Sharding spec sharding rng key across batches and replicating
        all other optimizer variables
    """
    def shard_optimizer_component(state_component):
        if isinstance(state_component, ScaleByLowRankOrthogonalUpdateState):
            return ScaleByLowRankOrthogonalUpdateState(
                    step=replicated,
                    shape_info=replicated,
                    momentum=jax.tree.map(lambda _: replicated, state_component.momentum),
                    key=jax.tree.map(lambda _: sharded, state_component.key),
                    )
        else:
            return jax.tree.map(lambda _: replicated, state_component)

    return jax.tree.map(
            shard_optimizer_component,
            optimizer_state,
            is_leaf=lambda x: (
                isinstance(x, ScaleByLowRankOrthogonalUpdateState) or
                (
                    hasattr(x, '_fields') and
                    not isinstance(x, ScaleByLowRankOrthogonalUpdateState)
                    )
                )
            )


def train_step(workload,
               opt_update_fn,
               model_state,
               optimizer_state,
               current_param_container,
               batch,
               rng,
               grad_clip,
               label_smoothing,
               dropout_rate,
               ):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True,
        dropout_rate=dropout_rate,
        )
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  loss = summed_loss / n_valid_examples
  grad = jax.tree.map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree.map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
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
    train_state: Optional[Dict[str, Any]] = None) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results
  del hyperparameters

  hyperparameters = HPARAMS

  optimizer_state, opt_update_fn = optimizer_state
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters['label_smoothing']
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters['grad_clip']
  else:
    grad_clip = None
  dropout_rate = hyperparameters['dropout_rate']
  
  # mesh = jax.sharding.Mesh(jax.devices(), ('batch'))
  replicated = jax_sharding_utils.get_replicate_sharding()
  sharded = (
          jax_sharding_utils.get_batch_dim_sharding()
          )
  optimizer_state_sharding = create_ademamix_sharding_from_names(
          optimizer_state=optimizer_state,
          params_tree=current_param_container,
          replicated=replicated,
          sharded=sharded
          )
  arg_shardings = (
          replicated, #model_state
          optimizer_state_sharding, #optimizer_state # change to optimizer sharding eventually
          replicated, # current_param_container
          sharded, # batch
          replicated, # per_device_rngs
          replicated, # grad_clip
          replicated, #label_smoothing
          replicated, #dropout_rate
          )
  out_shardings = (
          optimizer_state_sharding, # new_optimizer_state # maybe sharded eventually
          replicated, # updated_params
          replicated, # new_model_state
          replicated, # loss
          replicated, # grad_norm
          )
  jitted_train_step = jax.jit(
          train_step,
          static_argnums=(0, 1),
          donate_argnums=(2, 3, 4),
          in_shardings=arg_shardings,
          out_shardings=out_shardings,
          )
  outputs = jitted_train_step(workload,
                              opt_update_fn,
                              model_state,
                              optimizer_state,
                              current_param_container,
                              batch,
                              rng,
                              grad_clip,
                              label_smoothing,
                              dropout_rate,
                              )
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss,
            'grad_norm': grad_norm,
        }, global_step)
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
        return 128
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
    T = workload.step_hint
    f_b3 = beta3_scheduler(b3, beta_start=b1, warmup=T)
    f_a = alpha_scheduler(alpha, alpha_start=0, warmup=T)
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
    return optimizer_state, opt_update_fn

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
