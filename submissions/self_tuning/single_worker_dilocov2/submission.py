"""Submission file for Single-Worker DiLoCo optimizer with warmup+cosine LR in JAX with configurable CPU Offloading."""

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algoperf import jax_sharding_utils, spec 

# Fixed hyperparameters matching the Single-Worker DiLoCo init2winit config
HPARAMS = {
    'dropout_rate': 0.0,
    'learning_rate': 0.000955,
    'warmup_factor': 0.0,
    'sync_period': 75,
    'beta1': 0.99,
    'beta2': 0.995,
    'epsilon': 1e-8,
    'weight_decay': 0.2424,
    'outer_lr': 0.3006, 
    'outer_momentum': 0.7675,
    'cpu_offload': True,  # Toggle to False to keep slow_params & nesterov_b continuously on the TPU
}

# ==============================================================================
# DATA STRUCTURES & LOGIC COMPONENTS
# ==============================================================================

class CpuOffloaded:
  """Marker wrapper for arrays that should remain on CPU."""
  def __init__(self, array):
    self.array = array

  @property
  def shape(self):
    return self.array.shape

  @property
  def dtype(self):
    return self.array.dtype

  def __repr__(self):
    return f'CpuOffloaded(shape={self.shape}, dtype={self.dtype})'

# Register globally within the framework
jax.tree_util.register_pytree_node(
    CpuOffloaded,
    flatten_func=lambda x: ((x.array,), None),
    unflatten_func=lambda aux, children: CpuOffloaded(children[0])
)

class SingleWorkerDiLoCoState(NamedTuple):
  """State for Single-Worker DiLoCo."""
  adamw_m: Any
  adamw_v: Any
  slow_params: Any
  nesterov_b: Any
  global_inner_step: jax.Array
  fast_params: Any


def get_lr_schedule_fn(step_hint: int, hyperparameters: Dict[str, Any]):
  """Creates a standard Cosine schedule with optional warmup."""
  warmup_steps = int(hyperparameters['warmup_factor'] * step_hint)
  
  if warmup_steps > 0:
      warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=hyperparameters['learning_rate'],
        transition_steps=warmup_steps,
      )
  else:
      warmup_fn = None
      
  cosine_steps = max(step_hint - warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
    init_value=hyperparameters['learning_rate'], decay_steps=cosine_steps
  )
  
  if warmup_fn is not None:
      return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
      )
  return cosine_fn


def inner_train_step(
  workload,
  current_param_container,
  model_state,
  adamw_m,
  adamw_v,
  global_inner_step,
  batch,
  rng,
  lr,
  beta1,
  beta2,
  epsilon,
  weight_decay,
  dropout_rate,
  label_smoothing,
):
  """JIT-compiled inner step: gradient computation + AdamW update."""
  def _loss_fn(params):
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
      label_smoothing=label_smoothing,
    )
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
    current_param_container
  )
  grad = jax.tree_util.tree_map(lambda x: x / n_valid_examples, grad)

  global_inner_step = global_inner_step + 1

  # Compute Inner AdamW Update Iterations
  adamw_m = jax.tree_util.tree_map(
      lambda m, g: beta1 * m + (1 - beta1) * g, adamw_m, grad
  )
  adamw_v = jax.tree_util.tree_map(
      lambda v, g: beta2 * v + (1 - beta2) * g**2, adamw_v, grad
  )

  bc1 = 1 - beta1**global_inner_step
  bc2 = 1 - beta2**global_inner_step
  m_hat = jax.tree_util.tree_map(lambda m: m / bc1, adamw_m)
  v_hat = jax.tree_util.tree_map(lambda v: v / bc2, adamw_v)

  updated_params = jax.tree_util.tree_map(
      lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + epsilon) + weight_decay * p),
      current_param_container,
      m_hat,
      v_hat,
  )

  return updated_params, new_model_state, adamw_m, adamw_v, global_inner_step


def outer_step(
  fast_params,
  slow_params,
  nesterov_b,
  outer_lr,
  outer_momentum,
):
  """JIT-compiled outer step: Nesterov SGD update on the slow parameters."""
  pseudo_grad = jax.tree_util.tree_map(lambda s, f: s - f, slow_params, fast_params)

  # Momentum: b_t = mu * b_{t-1} + pseudo_grad
  nesterov_b = jax.tree_util.tree_map(
      lambda b, pg: outer_momentum * b + pg, nesterov_b, pseudo_grad
  )

  # Update slow params: theta_slow -= eta * (pseudo_grad + mu * b_t)
  slow_params = jax.tree_util.tree_map(
      lambda s, pg, b: s - outer_lr * (pg + outer_momentum * b),
      slow_params,
      pseudo_grad,
      nesterov_b,
  )

  # Fast parameters align symmetrically with slow parameters directly following communication phase
  new_fast = slow_params 
  return new_fast, slow_params, nesterov_b


# ==============================================================================
# ALGOPERF API METHODS
# ==============================================================================

def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates a Single-Worker DiLoCo optimizer configuration with conditional offloading mechanics."""
  del workload
  del model_state
  del rng
  del hyperparameters

  # Initialize AdamW moments directly mapped to processing device (TPU) memory
  adamw_m = jax.tree_util.tree_map(jnp.zeros_like, model_params)
  adamw_v = jax.tree_util.tree_map(jnp.zeros_like, model_params)

  if HPARAMS['cpu_offload']:
    # Initialize slow_params and nesterov_b safely offloaded to Host CPU using CpuOffloaded wrapper
    params_cpu = jax.device_get(model_params)
    slow_params = jax.tree_util.tree_map(
        lambda x: CpuOffloaded(np.copy(x)), params_cpu
    )
    nesterov_b = jax.tree_util.tree_map(
        lambda x: CpuOffloaded(np.zeros_like(x)), params_cpu
    )
  else:
    # Everything is retained natively on the acceleration hardware (TPU)
    slow_params = jax.tree_util.tree_map(jnp.array, model_params)
    nesterov_b = jax.tree_util.tree_map(jnp.zeros_like, model_params)

  return SingleWorkerDiLoCoState(
      adamw_m=adamw_m,
      adamw_v=adamw_v,
      slow_params=slow_params,
      nesterov_b=nesterov_b,
      global_inner_step=jnp.array(0, dtype=jnp.int32),
      fast_params=None,
  ), None


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
  cpu_offload = hyperparameters['cpu_offload']
  opt_state, _ = optimizer_state
  
  # Seamlessly swap fast_params back into scope upon resuming training after validation runs
  if opt_state.fast_params is not None:
    current_param_container = opt_state.fast_params

  lr_schedule_fn = get_lr_schedule_fn(workload.step_hint, hyperparameters)
  lr = lr_schedule_fn(global_step)
  
  label_smoothing = hyperparameters.get('label_smoothing', 0.0)
  dropout_rate = hyperparameters['dropout_rate']

  replicated = jax_sharding_utils.get_replicate_sharding()
  sharded = jax_sharding_utils.get_batch_dim_sharding()

  arg_shardings = (
    replicated,  # current_param_container
    replicated,  # model_state
    replicated,  # adamw_m
    replicated,  # adamw_v
    replicated,  # global_inner_step
    sharded,     # batch
    replicated,  # rng
    replicated,  # lr
    replicated,  # beta1
    replicated,  # beta2
    replicated,  # epsilon
    replicated,  # weight_decay
    replicated,  # dropout_rate
    replicated,  # label_smoothing
  )

  out_shardings = (
    replicated,  # updated_params
    replicated,  # new_model_state
    replicated,  # adamw_m
    replicated,  # adamw_v
    replicated,  # global_inner_step
  )

  jitted_inner_step = jax.jit(
    inner_train_step,
    static_argnums=(0,),
    donate_argnums=(1, 2, 3, 4),
    in_shardings=arg_shardings,
    out_shardings=out_shardings,
  )

  # Fast / Inner Loop Dispatch on Hardware
  (
    new_fast,
    new_model_state,
    new_adamw_m,
    new_adamw_v,
    new_inner_step
  ) = jitted_inner_step(
    workload,
    current_param_container,
    model_state,
    opt_state.adamw_m,
    opt_state.adamw_v,
    opt_state.global_inner_step,
    batch,
    rng,
    lr,
    hyperparameters['beta1'],
    hyperparameters['beta2'],
    hyperparameters['epsilon'],
    hyperparameters['weight_decay'],
    dropout_rate,
    label_smoothing,
  )

  is_outer_step = ((global_step + 1) % hyperparameters['sync_period']) == 0

  # Slow / Outer Loop Dispatch & Parameter Re-Sync Protocol
  if is_outer_step:
    jitted_outer_step = jax.jit(
        outer_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(replicated, replicated, replicated, replicated, replicated),
        out_shardings=(replicated, replicated, replicated),
    )

    if cpu_offload:
      replicated = jax_sharding_utils.get_replicate_sharding()
      slow_params_tpu = jax.tree_util.tree_map(
          lambda x: jax.device_put(x.array, replicated),
          opt_state.slow_params,
          is_leaf=lambda x: isinstance(x, CpuOffloaded)
      )
      nesterov_b_tpu = jax.tree_util.tree_map(
          lambda x: jax.device_put(x.array, replicated),
          opt_state.nesterov_b,
          is_leaf=lambda x: isinstance(x, CpuOffloaded)
      )
    else:
      # Data is already local to the TPU context
      slow_params_tpu = opt_state.slow_params
      nesterov_b_tpu = opt_state.nesterov_b

    new_fast, new_slow_params_tpu, new_nesterov_b_tpu = jitted_outer_step(
        new_fast,
        slow_params_tpu,
        nesterov_b_tpu,
        hyperparameters['outer_lr'],
        hyperparameters['outer_momentum'],
    )

    if cpu_offload:
      # Transfer back TPU -> CPU and rewrap in CpuOffloaded[
      slow_params = jax.tree_util.tree_map(
          lambda x: CpuOffloaded(np.copy(jax.device_get(x))), new_slow_params_tpu
      )
      nesterov_b = jax.tree_util.tree_map(
          lambda x: CpuOffloaded(np.copy(jax.device_get(x))), new_nesterov_b_tpu
      )
    else:
      # Retain updated array addresses in accelerator framework memory
      slow_params = new_slow_params_tpu
      nesterov_b = new_nesterov_b_tpu
  else:
    slow_params = opt_state.slow_params
    nesterov_b = opt_state.nesterov_b

  new_optimizer_state = SingleWorkerDiLoCoState(
      adamw_m=new_adamw_m,
      adamw_v=new_adamw_v,
      slow_params=slow_params,
      nesterov_b=nesterov_b,
      global_inner_step=new_inner_step,
      fast_params=None,  # Reset validation pipeline pointer explicitly
  ), None

  return new_optimizer_state, new_fast, new_model_state


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
  """Yield slow_params for model evaluations to match EMA properties without breaking fast param tracks."""
  del workload
  del current_params_types
  del hyperparameters
  del loss_type
  del eval_results
  del global_step
  del rng
  opt_state, _ = optimizer_state

  # Safe keeping fast_params state array in case it triggers immediately before next phase iteration
  new_optimizer_state = SingleWorkerDiLoCoState(
      adamw_m=opt_state.adamw_m,
      adamw_v=opt_state.adamw_v,
      slow_params=opt_state.slow_params,
      nesterov_b=opt_state.nesterov_b,
      global_inner_step=opt_state.global_inner_step,
      fast_params=current_param_container,
  ), None

  if HPARAMS['cpu_offload']:
    # Unwrap from CpuOffloaded wrapper and reconstruct on TPU for accurate validation runs
    slow_params_tpu = jax.tree_util.tree_map(
        lambda x: jnp.array(x.array), 
        opt_state.slow_params,
        is_leaf=lambda x: isinstance(x, CpuOffloaded)
    )
  else:
    slow_params_tpu = opt_state.slow_params

  return new_optimizer_state, slow_params_tpu, model_state


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
  elif workload_name == 'finewebedu_lm':
    return 64
  elif workload_name == 'mnist':
    return 16
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
  return next(input_queue)
