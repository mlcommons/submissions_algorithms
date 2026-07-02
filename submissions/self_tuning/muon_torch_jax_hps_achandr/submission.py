""""
Muon PyTorch DDP submission.
Backup optimizer: NadamW.
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

from submissions_algorithms.submissions.self_tuning.muon_torch_jax_hps.muon import MuonDataParallel, split_params_muon_adam

USE_PYTORCH_DDP = pytorch_setup()[0]

# Best Muon PyTorch Hyperparameters
HPARAMS = {
  "muon_learning_rate": 0.0001357495179 * 28.61130298,
  "muon_weight_decay": 0.02397563985,
  "muon_beta": 0.9560745273,
  "muon_adjust_lr": "spectral_norm",
  "muon_nesterov": False,
  "muon_ns_steps": 5,
  "muon_ns_eps": 1e-07,
  "nadamw_learning_rate": 0.0001357495179,
  "nadamw_weight_decay": 0.01275028385,
  "nadamw_beta1": 0.96586744,
  "nadamw_beta2": 0.8629173173,
  "nadamw_eps": 1e-08,
  "dropout_rate": 0.0,
  "label_smoothing": 0.0,
  "warmup_factor": 0.0,
  "step_reduce": 1.0
}

HPARAMS = SimpleNamespace(**HPARAMS)


def _pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
  warmup_steps = int(hyperparameters.warmup_factor * step_hint)
  total_steps = int(step_hint * getattr(hyperparameters, "step_reduce", 1.0))
  
  # If warmup is 0, return CosineAnnealingLR directly to bypass the SequentialLR milestone-0 collapse
  if warmup_steps == 0:
    return CosineAnnealingLR(optimizer, T_max=total_steps)
    
  warmup = LinearLR(
    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
  )
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
    'muon': MuonDataParallel(
      muon_params,
      lr=hyperparameters.muon_learning_rate,
      weight_decay=hyperparameters.muon_weight_decay,
      beta=hyperparameters.muon_beta,
      nesterov=hyperparameters.muon_nesterov,
      ns_steps=hyperparameters.muon_ns_steps,
      ns_eps=hyperparameters.muon_ns_eps,
      adjust_lr=hyperparameters.muon_adjust_lr,
    ),
    'nadamw': torch.optim.NAdam(
      adam_params,
      lr=hyperparameters.nadamw_learning_rate,
      weight_decay=hyperparameters.nadamw_weight_decay,
      betas=(hyperparameters.nadamw_beta1, hyperparameters.nadamw_beta2),
      eps=hyperparameters.nadamw_eps,
      decoupled_weight_decay=True,
      foreach=True
    ),
  }

  # One scheduler per optimizer
  optimizer_state['muon_scheduler'] = _pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['muon']
  )
  optimizer_state['nadamw_scheduler'] = _pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['nadamw']
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
  optimizer_state['nadamw'].zero_grad()

  # Skip all_reduce in backward pass:
  current_model.require_backward_grad_sync=False

  # Fwd pass wrapped in Autocast (enables GPU bfloat16 hardware acceleration)
  with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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

  # Compute grads, but do not AllReduce them.
  loss.backward()

  # DIAGNOSTICS: Print state details for the first 5 steps
  if global_step <= 5:
    muon_lr = optimizer_state['muon'].param_groups[0]['lr']
    nadamw_lr = optimizer_state['nadamw'].param_groups[0]['lr']
    
    muon_grads = [p.grad for p in optimizer_state['muon'].param_groups[0]['params'] if p.grad is not None]
    nadamw_grads = [p.grad for p in optimizer_state['nadamw'].param_groups[0]['params'] if p.grad is not None]
    
    muon_grad_norm = sum(g.norm().float()**2 for g in muon_grads).item()**0.5 if muon_grads else 0.0
    nadamw_grad_norm = sum(g.norm().float()**2 for g in nadamw_grads).item()**0.5 if nadamw_grads else 0.0
    
    logging.info(
        f"\n=== DIAGNOSTICS [Step {global_step}] ===\n"
        f"  • Muon LR:      {muon_lr:.12f}  |  Grad Norm: {muon_grad_norm:.6f} (from {len(muon_grads)} gradients)\n"
        f"  • NAdamW LR:    {nadamw_lr:.12f}  |  Grad Norm: {nadamw_grad_norm:.6f} (from {len(nadamw_grads)} gradients)\n"
        f"  • Muon Params:  {len(optimizer_state['muon'].param_groups[0]['params'])} total tensors\n"
        f"  • NAdamW Params:{len(optimizer_state['nadamw'].param_groups[0]['params'])} total tensors\n"
        f"========================================\n"
    )

  # Manually all-reduce AdamW grads
  for group in optimizer_state['nadamw'].param_groups:
    for p in group['params']:
      if p.grad is not None:
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

  if grad_clip is not None and grad_clip != 0:
    raise NotImplementedError('Grad clipping not supported by MuonDataParallel.')

  optimizer_state['muon'].step()
  optimizer_state['nadamw'].step()
  optimizer_state['muon_scheduler'].step()
  optimizer_state['nadamw_scheduler'].step()

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
