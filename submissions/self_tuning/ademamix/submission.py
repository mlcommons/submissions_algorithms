"""AlgoPerf AdEMAMix submission built on Optax."""
import functools
from typing import (
        Any,
        Dict,
        Iterator,
        List,
        Optional,
        Tuple,
        )

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils

from algoperf import spec, jax_sharding_utils


HPARAMS = {
        'ademamix_variant': 'simplified',
        'alpha': 8.0,
        'warmup_factor': 0.02,
        'beta3_warmup': 500e3,
        'alpha_warmup': 500e3,
        'learning_rate': 2e-3,
        'one_minus_beta1': 0.2,
        'beta2': 0.995,
        'beta3': 0.9995,
        'eps': 1e-8,
        'eps_root': 0.0,
        'weight_decay': 0.1,
        'grad_clip': 0.5,
        'dropout_rate': 0.1,
        }
        
_GRAD_CLIP_EPS = 1e-6

def lr_scheduler(learning_rate, warmup_factor, total_steps):
    warmup_steps = int(warmup_factor * total_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    return optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 1e-10,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + cosine_steps,
            end_value=0.0
            )

def alpha_scheduler(alpha, warmup=0):
    warmup_fn = optax.linear_schedule(init_value=0, end_value=alpha, transition_steps=warmup)
    constant_fn = optax.constant_schedule(alpha)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[warmup])
    return schedule_fn


def beta3_scheduler(beta3, beta1=0, warmup=0):

    def f(beta):
        return jnp.log(0.5)/jnp.log(beta)-1

    def f_inv(t):
        return jnp.power(0.5, 1/(t+1))

    def warmup_fn(step):
        frac = 1 - step / warmup
        return f_inv( frac * f(beta1) + (1 - frac) * f(beta3))

    constant_fn = optax.constant_schedule(beta3)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[warmup])
    return schedule_fn

def build_ademamix_optimizer(
    lr,
    variant='simplified',
    b1=0.9,
    b2=0.999,
    b3=0.9999,
    alpha=5.0,
    b3_scheduler=None,
    alpha_scheduler=None,
    eps=1e-8,
    eps_root=0.0,
    weight_decay=0.0,
    mask=None,
):
  if variant == 'simplified':
    return optax.contrib.simplified_ademamix(
        learning_rate=lr,
        b1=b1,
        b2=b2,
        alpha=alpha_scheduler if alpha_scheduler is not None else alpha,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        mask=mask,
    )
  if variant == 'full':
    return optax.contrib.ademamix(
        learning_rate=lr,
        b1=b1,
        b2=b2,
        b3=b3_scheduler if b3_scheduler is not None else b3,
        alpha=alpha_scheduler if alpha_scheduler is not None else alpha,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        mask=mask,
    )
  raise ValueError(f'Unsupported ademamix variant: {variant}')


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

  
replicated = jax_sharding_utils.get_replicate_sharding()
sharded = jax_sharding_utils.get_batch_dim_sharding()
arg_shardings = (
        replicated,  # model_state
        replicated,  # optimizer_state
        replicated,  # current_param_container
        sharded,     # batch
        replicated,  # per_device_rngs
        replicated,  # grad_clip
        replicated,  # label_smoothing
        replicated,  # dropout_rate
        )
out_shardings = (
        replicated, # new_optimizer_state
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
    one_minus_beta1 = HPARAMS['one_minus_beta1']
    b1 = 1.0 - one_minus_beta1
    b2 = HPARAMS['beta2']
    b3 = HPARAMS['beta3']
    alpha = HPARAMS['alpha']
    variant = HPARAMS['ademamix_variant']
    warmup_factor = HPARAMS['warmup_factor']
    beta3_warmup = HPARAMS['beta3_warmup']
    alpha_warmup = HPARAMS['alpha_warmup']
    T = workload.step_hint
    f_b3 = beta3_scheduler(b3, beta1=b1, warmup=beta3_warmup)
    f_a = alpha_scheduler(alpha, warmup=alpha_warmup)
    f_lr = lr_scheduler(lr, warmup_factor, T)
    weight_decay = HPARAMS['weight_decay']
    optimizer = build_ademamix_optimizer(
        lr=f_lr,
        variant=variant,
        b1=b1,
        b2=b2,
        b3=b3,
        alpha=alpha,
        b3_scheduler=f_b3,
        alpha_scheduler=f_a,
        weight_decay=weight_decay,
    )
    opt_init_fn = optimizer.init
    opt_update_fn = optimizer.update
    optimizer_state = opt_init_fn(params_zeros_like)
    return optimizer_state, opt_update_fn

if __name__ == "__main__": # dummy test
   
    def f(x): return jnp.sum(x ** 2)  # simple quadratic function

    alpha = 8.0
    one_minus_beta1 = 0.1
    b1 = 1.0 - one_minus_beta1
    b2, b3 = 0.999, 0.9999

    f_a = alpha_scheduler(alpha, warmup=10)
    f_b3 = beta3_scheduler(b3, beta1=b1, warmup=10)

    solver = build_ademamix_optimizer(
        lr=0.01,
        variant='full',
        b1=b1,
        b2=b2,
        b3=b3,
        alpha=alpha,
        b3_scheduler=f_b3,
        alpha_scheduler=f_a,
        weight_decay=0.01,
    )
    
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
