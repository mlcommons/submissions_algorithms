""""
Muon PyTorch DDP submission.
Backup optimizer: AdamW.
LR schedule: linear warmup + cosine decay, we support shorter schedules via `step_reduce` hyperparameter.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup

from submissions_algorithms.submissions.self_tuning.muon_torch_replicated_torch_hps.muon import MuonSingleDevice, split_params_muon_adam

USE_PYTORCH_DDP = pytorch_setup()[0]

# Best Muon PyTorch Hyperparameters
HPARAMS = {
  "learning_rate": 0.01,
  "muon_weight_decay": 0.0,
  "muon_beta": 0.9,
  "muon_adjust_lr": "spectral_norm",
  "muon_nesterov": True,
  "muon_ns_steps": 5,
  "muon_ns_eps": 1e-07,
  "adamw_weight_decay": 0.0,
  "adamw_beta1": 0.9,
  "adamw_beta2": 0.999,
  "adamw_eps": 1e-08,
  "dropout_rate": 0.1,
  "label_smoothing": 0.1,
  "warmup_factor": 0.05,
  "step_reduce": 1.0
}
HPARAMS = SimpleNamespace(**HPARAMS)


def _pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
  warmup_steps = int(hyperparameters.warmup_factor * step_hint)
  warmup = LinearLR(
    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
  )
  total_steps = int(step_hint * getattr(hyperparameters, "step_reduce", 1.0))
  decay_steps = max(1, total_steps - warmup_steps)
  cosine_decay = CosineAnnealingLR(optimizer, T_max=decay_steps)
  return SequentialLR(
    optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps]
  )


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates a Muon optimizer and a learning rate schedule."""
  del model_state
  del rng

  hyperparameters = HPARAMS

  muon_params, adam_params = split_params_muon_adam(model_params)

  optimizer_state = {
    'muon': MuonSingleDevice(
      muon_params,
      lr=hyperparameters.learning_rate,  # shared
      weight_decay=hyperparameters.muon_weight_decay,
      beta=hyperparameters.muon_beta,
      nesterov=hyperparameters.muon_nesterov,
      ns_steps=hyperparameters.muon_ns_steps,
      ns_eps=hyperparameters.muon_ns_eps,
      adjust_lr=hyperparameters.muon_adjust_lr,
    ),
    'adamw': torch.optim.AdamW(
      adam_params,
      lr=hyperparameters.learning_rate,  # shared
      weight_decay=hyperparameters.adamw_weight_decay,
      betas=(hyperparameters.adamw_beta1, hyperparameters.adamw_beta2),
      eps=hyperparameters.adamw_eps,
      fused=True
    ),
  }

  # One scheduler per optimizer
  optimizer_state['muon_scheduler'] = _pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['muon']
  )
  optimizer_state['adamw_scheduler'] = _pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['adamw']
  )

  return optimizer_state


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
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results

  hyperparameters = HPARAMS

  reduced_steps = int(workload.step_hint * getattr(hyperparameters, "step_reduce", 1.0))
  if global_step >= reduced_steps:
    raise spec.TrainingCompleteError(
      f"Stopping at step {global_step}/{reduced_steps} "
      f"(step_reduce={hyperparameters.step_reduce})"
    )

  current_model = current_param_container
  current_model.train()
  optimizer_state['muon'].zero_grad()
  optimizer_state['adamw'].zero_grad()

  # Fwd pass
  logits_batch, new_model_state = workload.model_fn(
    params=current_model,
    augmented_and_preprocessed_input_batch=batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.TRAIN,
    rng=rng,
    update_batch_norm=True,
    dropout_rate=hyperparameters.dropout_rate,
  )

  # Bwd pass
  label_smoothing = (
    hyperparameters.label_smoothing
    if hasattr(hyperparameters, 'label_smoothing')
    else 0.0
  )
  grad_clip = getattr(hyperparameters, 'grad_clip', None)

  loss_dict = workload.loss_fn(
    label_batch=batch['targets'],
    logits_batch=logits_batch,
    mask_batch=batch.get('weights'),
    label_smoothing=label_smoothing,
  )
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  # AllReduce grads
  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )

  optimizer_state['muon'].step()
  optimizer_state['adamw'].step()
  optimizer_state['muon_scheduler'].step()
  optimizer_state['adamw_scheduler'].step()

  # # Log training metrics - loss, grad_norm, batch_size.
  # if global_step <= 100 or global_step % 50 == 0:
  #   with torch.no_grad():
  #     parameters = [p for p in current_model.parameters() if p.grad is not None]
  #     grad_norm = torch.norm(
  #       torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
  #     )
  #   if workload.metrics_logger is not None:
  #     workload.metrics_logger.append_scalar_metrics(
  #       {
  #         'loss': loss.item(),
  #         'grad_norm': grad_norm.item(),
  #       },
  #       global_step,
  #     )
  #   logging.info(
  #     '%d) loss = %0.3f, grad_norm = %0.3f',
  #     global_step,
  #     loss.item(),
  #     grad_norm.item(),
  #   )

  return (optimizer_state, current_param_container, new_model_state)


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
  elif workload_name == 'finewebedu_lm':
    return 64
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
