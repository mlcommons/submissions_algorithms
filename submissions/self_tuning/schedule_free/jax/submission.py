"""Submission file for an Schedule Free AdamW optimizer in Jax."""

from typing import Dict, Iterator, List, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from optax.contrib import schedule_free_adamw

from algoperf import spec
from jax.sharding import NamedSharding, PartitionSpec as P


_GRAD_CLIP_EPS = 1e-6

HPARAMS = {
  'dropout_rate': 0.1,
  'learning_rate': 0.0025,
  'one_minus_beta1': 0.1,
  'beta2': 0.9955159689799007,
  'weight_decay': 0.08121616522670176,
  'warmup_factor': 0.02,
  'weight_lr_power': 2,
  'label_smoothing': 0.2,
  'r': 0.75,
  'eps': 1e-8,
}

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
) -> Tuple[spec.OptimizerState, spec.ParameterContainer, spec.ModelAuxiliaryState]:
    """Converts y (training params) to x (eval params) using the SF state."""
    (state, _), opt_update_fn = optimizer_state
    
    # Calculate x = (y - (1 - b1) * z) / b1
    params_for_eval = schedule_free_eval_params(state, current_param_container) # (current_param_container - (1 - state.b1) * state.z) / state.b1

    is_holding_x = jnp.array(1, dtype=jnp.int32)

    new_optimizer_state = ((state, is_holding_x), opt_update_fn)
    
    # We return params_for_eval x
    return new_optimizer_state, params_for_eval, model_state


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  model_params
  del model_state
  del rng

  opt_init_fn, opt_update_fn = schedule_free_adamw(
    learning_rate=HPARAMS['learning_rate'],
    warmup_steps=int(HPARAMS['warmup_factor'] * workload.step_hint * 0.75),
    b1=1.0 - HPARAMS['one_minus_beta1'],
    b2=HPARAMS['beta2'],
    eps=HPARAMS['eps'],
    weight_decay=HPARAMS['weight_decay'],
    weight_lr_power=HPARAMS['weight_lr_power'],
    # state_dtype=jnp.bfloat16
  )

  model_params = jax_utils.unreplicate(model_params)
  optimizer_state = opt_init_fn(model_params)
  is_holding_x = jnp.array(0, dtype=jnp.int32)

  return (optimizer_state, is_holding_x), opt_update_fn


def train_step(
  workload,
  opt_update_fn,
  model_state,
  optimizer_state,
  current_param_container,
  batch,
  rng,
  grad_clip,
  label_smoothing,
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

  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
    sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad))
  )

  # Extract the leaves of the pytree
  # leaves = jax.tree_util.tree_leaves(grad)

  # Count the total number of elements in all leaves
  # total_size = sum(jnp.size(leaf) for leaf in leaves)

  # jax.debug.print('GRAD NORM {}', grad_norm)
  # jax.debug.print('NUM PARAMS {}', total_size)

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

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
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  (optimizer_state, is_holding_x), opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

  if is_holding_x > 0:
    beta1 = 1.0 - HPARAMS['one_minus_beta1']
    z = optimizer_state.z
        
    # restore Y: y = (1 - b) * z + b * x
    current_param_container = jax.tree.map(
        lambda x_leaf, z_leaf: (1 - beta1) * z_leaf + beta1 * x_leaf,
        current_param_container, # x
        z
    )

  # Set up mesh and sharding
  mesh = jax.sharding.Mesh(jax.devices(), ('batch'))
  replicated = NamedSharding(mesh, P())  # No partitioning
  sharded = NamedSharding(mesh, P('batch'))  # Partition along batch dimension
  
  
  jitted_train_step = jax.jit(
      train_step,
      static_argnums=(0, 1),
      donate_argnums=(2, 3, 4),
      in_shardings=(
        # workload is static
        # opt_update_fn is static
        replicated,  # model_state
        replicated,  # optimizer_state
        replicated,  # current_param_container
        sharded,  # batch
        replicated,  # rng
        replicated,  # grad_clip
        replicated,  # label_smoothing
      ),
      out_shardings=(
        replicated,  # new_optimizer_state
        replicated,  # updated_params
        replicated,  # new_model_state
        replicated,  # loss
        replicated,  # grad_norm
      ),
    )
    # print(batch)
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = (
      jitted_train_step(
        workload,
        opt_update_fn,
        model_state,
        optimizer_state,
        current_param_container,
        batch,
        rng,
        grad_clip,
        label_smoothing,
      )
    )

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    workload.metrics_logger.append_scalar_metrics(
      {
        'loss': loss,
        'grad_norm': grad_norm,
      },
      global_step,
    )
  
  new_is_holding_x = jnp.array(0, dtype=jnp.int32)
  new_optimizer_state = ((new_optimizer_state, new_is_holding_x), opt_update_fn)

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
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
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
