"""
Submission file for Muon optimizer with warmup+cosine LR in PyTorch.
Vanilla DP implementation, each process orthogonalizes each parameter.

Which momentum implementation should we use?
(A) w/ dampening (1-beta):
    - KellerJordan/Muon: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L35
    - KellerJordan/modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt/blob/1b51e26d304f647c7c12201b3f1513ee5a429ec4/train_gpt.py#L197
(B) w/out dampening:
    - KellerJordan/cifar10: https://github.com/KellerJordan/cifar10-airbench/blob/0e6f9614572d7e8e3c259905aebc7196f91d5d79/research/clean_muon.py#L91
    - original KellerJordan/modded-nanogpt: https://github.com/microsoft/dion/blob/0360f9b0369603ecfa19de5128f56c983f1ac7d9/dion/muon_reference.py#L323
    - MoonShootAI: https://arxiv.org/pdf/2502.16982
    - Dion: https://github.com/microsoft/dion/tree/main
For now we stick to the original one, with no dampening,
but we should enable control on this with an additional hyperparameter.

TODO:
- check for dropout correctness
- how should we handle weight decay?
- add signum (Scion) backup (and perhaps lion?)
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


@torch.compile()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
  """
  Newton-Schulz iteration to approximally orthogonalize G.
  5-th order odd polynomial to approximate sign(x) on [-1,1],
  pushing singlular values to {+1,-1}.

  M = U @ S @ V.T
  sign(M) = U @ sign(S) @ V.T, odd matrix polynomial commutes with SVD
  sign(x) ~= a*x + b*x^3 + c*x^5, x in [-1,1]
  """
  assert len(G.shape) == 2, 'Expected 2D tensor'
  a, b, c = 3.4445, -4.7750, 2.0315
  X = G.bfloat16()
  if G.size(0) > G.size(1):
    X = X.T

  # Ensure spectral norm is at most 1.
  # Ortho(cX)=Ortho(X), so we can normalize by ||X||_2 <= ||X||_F
  X /= X.norm() + eps

  # NS iterations
  for _ in range(steps):
    A = X @ X.T
    B = b * A + c * (A @ A)
    X = a * X + B @ X

  if G.size(0) > G.size(1):
    X = X.T
  return X


class MuonVanilla(torch.optim.Optimizer):
  r"""Muon optimizer - Momentum Orthogonalized by Newton-Schulz."""

  def __init__(
    self,
    param_groups,
    lr=0.02,
    weight_decay=0.0,
    muon_beta=0.95,
    muon_nesterov=True,
    muon_ns_steps=5,
    muon_ns_eps=1.0e-7,
    adamw_beta1=0.9,
    adamw_beta2=0.999,
    adamw_eps=1e-8,
  ):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay: {weight_decay}')

    if not 0.0 <= muon_beta < 1.0:
      raise ValueError(f'Invalid muon_beta parameter: {muon_beta}')
    if muon_nesterov not in [True, False]:
      raise ValueError(f'Invalid muon_nesterov parameter: {muon_nesterov}')
    if not 0 < muon_ns_steps:
      raise ValueError(f'Invalid ns_steps parameter: {muon_ns_steps}')
    if not 0.0 <= muon_ns_eps:
      raise ValueError(f'Invalid muon_ns_eps parameter: {muon_ns_eps}')

    if not 0.0 <= adamw_beta1 < 1.0:
      raise ValueError(f'Invalid adamw_beta1 parameter: {adamw_beta1}')
    if not 0.0 <= adamw_beta2 < 1.0:
      raise ValueError(f'Invalid adamw_beta2 parameter: {adamw_beta2}')
    if not 0.0 <= adamw_eps:
      raise ValueError(f'Invalid adamw_eps: {adamw_eps}')

    # Manually set values that were not specified in param_groups
    for group in param_groups:
      group['lr'] = group.get('lr', lr)
      group['weight_decay'] = group.get('weight_decay', weight_decay)
      algo = group.get('algorithm')
      if algo == 'muon':
        group['beta'] = group.get('beta', muon_beta)
        group['nesterov'] = group.get('nesterov', muon_nesterov)
        group['ns_steps'] = group.get('ns_steps', muon_ns_steps)
        group['ns_eps'] = group.get('ns_eps', muon_ns_eps)
      elif algo == 'adamw':
        group['betas'] = group.get('betas', (adamw_beta1, adamw_beta2))
        group['eps'] = group.get('eps', adamw_eps)
      else:
        raise ValueError(f'Unsupported algorithm type: {algo}.')

    super().__init__(param_groups, dict())

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      wd = group['weight_decay']
      algo = group['algorithm']
      for p in group['params']:
        if p.grad is None:
          continue
        g = p.grad
        state = self.state[p]

        if algo == 'muon':
          beta = group['beta']
          nesterov = group['nesterov']
          ns_steps = group['ns_steps']
          ns_eps = group['ns_eps']

          if len(state) == 0:
            state['m'] = torch.zeros_like(p)

          state['m'].mul_(beta).add_(g)  # no dampening
          if nesterov:
            g = g.add(state['m'], alpha=beta)
          else:
            g = state['m']
          g = g.reshape(g.size(0), -1)  # flatten trailing dims (3D, 4D)
          g = zeropower_via_newtonschulz5(g, steps=ns_steps, eps=ns_eps)
          g = g.view(p.shape)  # restore original shape
          # Adjust from spectral norm 1 to RMS operator norm 1 https://arxiv.org/abs/2310.17813
          g *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5
          p.mul_(1 - lr * wd)
          p.add_(g, alpha=-lr)

        elif algo == 'adamw':
          beta1, beta2 = group['betas']
          eps = group['eps']

          if len(state) == 0:
            state['m'] = torch.zeros_like(p)
            state['v'] = torch.zeros_like(p)
            state['step'] = 0

          state['step'] += 1
          state['m'].lerp_(g, 1 - beta1)
          state['v'].lerp_(g.square(), 1 - beta2)
          mc = state['m'] / (1 - beta1 ** state['step'])
          vc = state['v'] / (1 - beta2 ** state['step'])
          update = mc / (vc.sqrt() + eps)

          p.mul_(1 - lr * wd)
          p.add_(update, alpha=-lr)

    return loss


def get_param_groups_muon(model, optim_backup):
  params = [p for p in model.parameters() if p.requires_grad]
  matrix_params = [p for p in params if p.ndim >= 2]
  non_matrix_params = [p for p in params if p.ndim < 2]

  # check + print
  named_params = [
    (n, p) for n, p in model.named_parameters() if p.requires_grad
  ]
  matrix_info = [f'{n} (ndim={p.ndim})' for n, p in named_params if p.ndim >= 2]
  non_matrix_info = [
    f'{n} (ndim={p.ndim})' for n, p in named_params if p.ndim < 2
  ]
  logging.info('matrix params:\n\t' + '\n\t'.join(matrix_info))
  logging.info('non-matrix params:\n\t' + '\n\t'.join(non_matrix_info))

  return [
    dict(params=matrix_params, algorithm='muon'),
    dict(params=non_matrix_params, algorithm=optim_backup),
  ]


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

  param_groups = get_param_groups_muon(
    model_params, optim_backup=hyperparameters.optim_backup
  )

  optimizer_state = {
    'optimizer': MuonVanilla(
      param_groups,
      lr=hyperparameters.learning_rate,  # shared
      weight_decay=hyperparameters.weight_decay,  # shared
      muon_beta=hyperparameters.muon_beta,
      muon_nesterov=hyperparameters.muon_nesterov,
      muon_ns_steps=hyperparameters.muon_ns_steps,
      muon_ns_eps=hyperparameters.muon_ns_eps,
      adamw_beta1=hyperparameters.adamw_beta1,
      adamw_beta2=hyperparameters.adamw_beta2,
      adamw_eps=hyperparameters.adamw_eps,
    ),
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup = LinearLR(
      optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
      optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps]
    )

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
    workload.step_hint, hyperparameters, optimizer_state['optimizer']
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
  optimizer_state['optimizer'].zero_grad()

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
    # TODO @nico: remove
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip
    )
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

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
    logging.info(
      '%d) loss = %0.3f, grad_norm = %0.3f',
      global_step,
      loss.item(),
      grad_norm.item(),
    )

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
