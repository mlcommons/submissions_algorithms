""""
MuonBucketed, with manual redcue_scatter for gradients,
see the corresponding algorithm in `muon_algos.py` for more details.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup

from reference_algorithms.muon.pytorch.muon_algos import MuonBucketed
from reference_algorithms.muon.pytorch.utils import _split_params_muon_adam

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
WORLD_SIZE = N_GPUS # single-node assumption


def _pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
  warmup_steps = int(hyperparameters.warmup_factor * step_hint)
  warmup = LinearLR(
    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
  )
  cosine_steps = max(step_hint - warmup_steps, 1)
  cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
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

  muon_params, adam_params = _split_params_muon_adam(model_params)

  optimizer_state = {
    'muon': MuonBucketed(
      muon_params,
      lr=hyperparameters.learning_rate,  # shared
      weight_decay=hyperparameters.weight_decay,  # shared
      beta=hyperparameters.muon_beta,
      dampening=hyperparameters.muon_dampening,
      nesterov=hyperparameters.muon_nesterov,
      ns_steps=hyperparameters.muon_ns_steps,
      ns_eps=hyperparameters.muon_ns_eps,
    ),
    'adamw': torch.optim.AdamW(
      adam_params,
      lr=hyperparameters.learning_rate,  # shared
      weight_decay=hyperparameters.weight_decay,  # shared
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
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

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
    # TODO @nico: is it ever the case that n_valid_examples differs across ranks?
    #             if not, we can safely remove this sync point.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  # Skip all_reduce in backward pass: compute grads locally on each rank, no sync
  current_model.require_backward_grad_sync=False
  loss.backward()
  
  # All-reduce AdamW grads
  for group in optimizer_state['adamw'].param_groups:
    for p in group['params']:
      if p.grad is not None:
        dist.all_reduce(p.grad)
        p.grad.div_(WORLD_SIZE)
  
  # Reduce-scatter Muon parameters
  # Space: O(largest_param) persistent (model + buffers);
  #        O(WORLD_SIZE × largest_param) transient during all_gather.
  # Comms: one per block (~#params/WORLD_SIZE)
  for group in optimizer_state['muon'].param_groups:
    for shape, params in group['bucket'].items():
      # Iterate over params in blocks of WORLD_SIZE
      for block_start in range(0, len(params), WORLD_SIZE):
        # Create (padded) block of gradients to reduce. Handles final truncated block when block_start + WORLD_SIZE exceeds len(params).
        grad_block = [
          (p.grad if p.grad is not None else torch.zeros_like(p))
          for p in params[block_start:block_start + WORLD_SIZE]
        ]
        pad = WORLD_SIZE - len(grad_block)
        if pad > 0:
          grad_block += [torch.zeros_like(grad_block[-1])] * pad

        grad_shard = torch.zeros_like(params[0]) # receiving buffer, same as indexing at [block_start + RANK]
        dist.reduce_scatter(grad_shard, grad_block) # reduce by sum
        grad_shard.div_(WORLD_SIZE) # avg

        # Assign averaged reduced grad to corresponding param
        if block_start + RANK < len(params):
            params[block_start + RANK].grad = grad_shard

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )
  optimizer_state['muon'].step()
  optimizer_state['adamw'].step()
  optimizer_state['muon_scheduler'].step()
  optimizer_state['adamw_scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 50 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
      )
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
        {
          'loss': loss.item(),
          'grad_norm': grad_norm.item(),
        },
        global_step,
      )
    # logging.info(
    #   '%d) loss = %0.3f, grad_norm = %0.3f',
    #   global_step,
    #   loss.item(),
    #   grad_norm.item(),
    # )

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
