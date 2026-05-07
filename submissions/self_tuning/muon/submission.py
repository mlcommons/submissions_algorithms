"""Submission file for a Muon optimizer with warmup+cosine LR in Jax."""

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
import jax.numpy as jnp
import optax

from algoperf import jax_sharding_utils, spec

_GRAD_CLIP_EPS = 1e-6

# Fixed hyperparameters matching the Muon init2winit config
HPARAMS = {
    'dropout_rate': 0.0,
    'learning_rate': 0.0001357495179,
    'warmup_factor': 0.0,
    'muon_beta': 0.9560745273,
    'muon_lr_multiplier': 28.61130298,
    'muon_weight_decay': 0.02397563985,
    'muon_nesterov': False,
    'pair_beta1': 0.96586744,
    'pair_beta2': 0.8629173173,
    'pair_eps': 1e-8,
    'pair_debias': True,
    'pair_weight_decay': 0.01275028385,
}

# ==============================================================================
# MUON OPTIMIZER COMPONENTS
# ==============================================================================

class MuonState(NamedTuple):
  """State for Muon."""
  count: chex.Array
  momentum: optax.Updates


def orthogonalize_via_newton_schulz(
    updates: jax.Array,
    newton_schulz_coeffs: jax.Array,
    newton_schulz_steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration."""
  if updates.ndim != 2:
    raise ValueError(f'Input must be 2D, got {updates.shape}')

  updates /= jnp.linalg.norm(updates) + eps

  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x

  transposed = False
  if updates.shape[0] > updates.shape[1]:
    updates = updates.T
    transposed = True

  orthogonalized_updates = jax.lax.fori_loop(
      0,
      newton_schulz_steps,
      lambda _, x: newton_schulz_iterator(x, newton_schulz_coeffs),
      updates,
  )

  if transposed:
    orthogonalized_updates = orthogonalized_updates.T

  fan_out = orthogonalized_updates.shape[1]
  fan_in = orthogonalized_updates.shape[0]

  scale_factor = jnp.maximum(1.0, jnp.sqrt(fan_out / fan_in))
  orthogonalized_updates *= scale_factor

  return orthogonalized_updates


def _default_reshape_fn(x):
  return x.reshape(x.shape[0], -1)


def scale_by_muon(
    beta: float = 0.95,
    weight_decay: float = 0.01,
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
    nesterov: bool = True,
    bias_correction: bool = False,
    reshape_fn=_default_reshape_fn,
) -> optax.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm."""
  ns_coeffs = jnp.asarray(ns_coeffs)

  def init_fn(params):
    momentum = jax.tree_util.tree_map(
        lambda p: jnp.zeros_like(p, jnp.float32), params
    )
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        momentum=momentum,
    )

  def update_fn(updates, state, params=None):
    running_momentum = _update_moment(updates, state.momentum, beta, 1)
    new_count = state.count + jnp.array(1, dtype=jnp.int32)

    if nesterov:
      momentum = _update_moment(updates, running_momentum, beta, 1)
    else:
      momentum = running_momentum

    if bias_correction:
      momentum = _bias_correction(momentum, beta, new_count)

    def _orthogonalize(x):
      orig_shape = x.shape
      if x.ndim > 2 and reshape_fn is not None:
        x = reshape_fn(x)
      x = orthogonalize_via_newton_schulz(x, ns_coeffs, ns_steps, eps)
      return x.reshape(orig_shape)

    scaled_orthogonalized_momentum = jax.tree_util.tree_map(
        _orthogonalize,
        momentum,
    )

    # Returns the update direction. Scaling by learning rate is chained later.
    new_updates = jax.tree_util.tree_map(
        lambda u, p: u + weight_decay * p,
        scaled_orthogonalized_momentum,
        params,
    )

    return new_updates, MuonState(
        count=new_count,
        momentum=running_momentum,
    )

  return optax.GradientTransformation(init_fn, update_fn)


# ==============================================================================
# NADAMW COMPONENTS (FOR THE 'PAIR' PARTITION)
# ==============================================================================

class ScaleByAdamState(NamedTuple):
  """State for the NAdam algorithm."""
  count: chex.Array
  mu: optax.Updates
  nu: optax.Updates


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_util.tree_map(
    lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
  )


def _bias_correction(moment, decay, count):
  """Perform bias correction."""
  beta = 1 - decay**count
  return jax.tree_util.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def scale_by_nadam(
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-8,
  eps_root: float = 0.0,
  debias: bool = True,
  power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm."""
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_util.tree_map(
      lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def nadamw(
  learning_rate: Union[float, optax.Schedule],
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-8,
  eps_root: float = 0.0,
  debias: bool = True,
  weight_decay: float = 0.0,
  weight_decay_mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
  return optax.chain(
    scale_by_nadam(b1, b2, eps, eps_root, debias),
    optax.add_decayed_weights(weight_decay, weight_decay_mask),
    scale_by_learning_rate(learning_rate),
  )


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
  """Creates a Muon/NAdamW partitioned optimizer and schedule."""
  del model_params
  del model_state
  del rng
  del hyperparameters

  hyperparameters = HPARAMS

  def jax_cosine_warmup(step_hint: int, hyperparameters):
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

  # Create schedule
  lr_schedule_fn = jax_cosine_warmup(workload.step_hint, hyperparameters)

  # Define Muon transform
  muon_tx = optax.chain(
      scale_by_muon(
          beta=hyperparameters['muon_beta'],
          weight_decay=hyperparameters['muon_weight_decay'],
          nesterov=hyperparameters['muon_nesterov']
      ),
      optax.scale(hyperparameters['muon_lr_multiplier']),
      scale_by_learning_rate(lr_schedule_fn)
  )

  # Define NAdamW transform
  pair_tx = nadamw(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters['pair_beta1'],
      b2=hyperparameters['pair_beta2'],
      eps=hyperparameters['pair_eps'],
      debias=hyperparameters['pair_debias'],
      weight_decay=hyperparameters['pair_weight_decay'],
  )

  # Partition function
  def muon_mask_fn(params):
    def _mask(path, x):
      path_str = jax.tree_util.keystr(path).lower()
      if x.ndim >= 2 and 'embed' not in path_str:
        return 'muon'
      return 'pair'
    return jax.tree_util.tree_map_with_path(_mask, params)

  # Initialize optimizer
  params_zeros_like = jax.tree_util.tree_map(
    lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes
  )
  
  labels = muon_mask_fn(params_zeros_like)
  optimizer = optax.multi_transform({'muon': muon_tx, 'pair': pair_tx}, labels)

  optimizer_state = optimizer.init(params_zeros_like)

  return optimizer_state, optimizer.update


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
  dropout_rate,
):
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
  loss = summed_loss / n_valid_examples
  grad = jax.tree_util.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
    sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad))
  )

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_util.tree_map(lambda x: x * grad_scaling_factor, grad)

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
  
  label_smoothing = hyperparameters.get('label_smoothing', 0.0)
  grad_clip = hyperparameters.get('grad_clip', None)
  dropout_rate = hyperparameters['dropout_rate']

  replicated = jax_sharding_utils.get_replicate_sharding()
  sharded = jax_sharding_utils.get_batch_dim_sharding()

  arg_shardings = (
    replicated,  # model_state
    replicated,  # optimizer_state
    replicated,  # current_param_container
    sharded,     # batch
    replicated,  # rng
    replicated,  # grad_clip
    replicated,  # label_smoothing
    replicated,  # dropout_rate
  )
  
  out_shardings = (
    replicated,  # new_optimizer_state
    replicated,  # updated_params
    replicated,  # new_model_state
    replicated,  # loss
    replicated,  # grad_norm
  )

  jitted_train_step = jax.jit(
    train_step,
    static_argnums=(0, 1),
    donate_argnums=(2, 3, 4),
    in_shardings=arg_shardings,
    out_shardings=out_shardings,
  )

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
      dropout_rate,
    )
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
  """Return (updated_optimizer_state, updated_params)."""
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
