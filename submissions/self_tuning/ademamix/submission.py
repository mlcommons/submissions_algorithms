from __future__ import annotations

import math
import collections
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed.nn as dist_nn
from absl import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.optimizer import Optimizer

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

USE_PYTORCH_DDP = pytorch_setup()[0]

# optimal parameters across all workloads
HPARAMS = {
  "learning_rate": 2e-3,
  "one_minus_beta1": 0.2,
  "beta2": 0.995,
  "beta3": 0.9995,
  "weight_decay": 0.1,
  "warmup_factor": 0.02,
  "alpha": 8,
  "beta3_warmup": 500e3,
  "alpha_warmup": 500e3,
  "grad_clip": 0.5,
}
HPARAMS = collections.namedtuple('Hyperparameters', HPARAMS.keys())(**HPARAMS)


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
  if step < warmup:
    a = step / float(warmup)
    return (1.0-a) * alpha_start + a * alpha_end
  return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

  def f(beta, eps=1e-8):
    return math.log(0.5)/math.log(beta+eps)-1

  def f_inv(t):
    return math.pow(0.5, 1/(t+1))

  if step < warmup:
    a = step / float(warmup)
    return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
  return beta_end


class AdEMAMix(Optimizer):
  r"""Implements the AdEMAMix algorithm.

  Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float, float], optional): coefficients used for computing
      running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
      corresponding to beta_1, beta_2, beta_3 in AdEMAMix
    alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
    beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
    alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
    eps (float, optional): term added to the denominator to improve
      numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay as in AdamW (default: 0)
  """

  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0, 
         beta3_warmup=None, alpha_warmup=None,  eps=1e-8,
         weight_decay=0):
    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= betas[2] < 1.0:
      raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
    if not 0.0 <= weight_decay:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    if not 0.0 <= alpha:
      raise ValueError("Invalid alpha value: {}".format(alpha))
    defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup, weight_decay=weight_decay)
    super(AdEMAMix, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(AdEMAMix, self).__setstate__(state)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      
      lr = group["lr"]
      wd = group["weight_decay"]
      eps = group["eps"]
      beta1, beta2, beta3_final = group["betas"]
      beta3_warmup = group["beta3_warmup"]
      alpha_final = group["alpha"]
      alpha_warmup = group["alpha_warmup"]
    
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        if grad.is_sparse:
          raise RuntimeError('AdEMAMix does not support sparse gradients.')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          if beta1 != 0.0: # save memory in case beta1 is 0.0
            state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          else: 
            state['exp_avg_fast'] = None
          state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        exp_avg_fast, exp_avg_slow, exp_avg_sq = state['exp_avg_fast'], state['exp_avg_slow'], state['exp_avg_sq']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Compute the effective alpha and beta3 in case warmup is used 
        if alpha_warmup is not None:
          alpha = linear_warmup_scheduler(state["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
        else:
          alpha = alpha_final
        
        if beta3_warmup is not None:
          beta3 = linear_hl_warmup_scheduler(state["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
        else:
          beta3 = beta3_final

        # Decay the first and second moment running average coefficient
        if beta1 != 0.0:
          exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
        else:
          exp_avg_fast = grad
        exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom

        # decay
        update.add_(p, alpha=wd)

        p.add_(-lr * update)

    return loss


def init_optimizer_state(workload: spec.Workload,
             model_params: spec.ParameterContainer,
             model_state: spec.ModelAuxiliaryState,
             hyperparameters: spec.Hyperparameters,
             rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Lion optimizer and a learning rate schedule."""
  del model_state
  del rng
  del hyperparameters

  hyperparameters = HPARAMS

  optimizer_state = {
    'optimizer':
      AdEMAMix(
        model_params.parameters(),
        lr=HPARAMS.learning_rate,
        betas=(1.0 - HPARAMS.one_minus_beta1,
             HPARAMS.beta2,
             HPARAMS.beta3),
        weight_decay=HPARAMS.weight_decay,
        alpha=HPARAMS.alpha,
        beta3_warmup= HPARAMS.beta3_warmup,
        alpha_warmup=HPARAMS.alpha_warmup)
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup = LinearLR(
      optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
      optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
    workload.step_hint, HPARAMS, optimizer_state['optimizer'])
  optimizer_state['hyperparameters'] = hyperparameters

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
  train_state: Optional[Dict[str, Any]] = None) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results
  del hyperparameters

  hyperparameters = HPARAMS

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
    params=current_model,
    augmented_and_preprocessed_input_batch=batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.TRAIN,
    rng=rng,
    update_batch_norm=True)

  label_smoothing = (
    hyperparameters.label_smoothing if hasattr(HPARAMS,
                         'label_smoothing') else 0.0)
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

  loss_dict = workload.loss_fn(
    label_batch=batch['targets'],
    logits_batch=logits_batch,
    mask_batch=batch.get('weights'),
    label_smoothing=label_smoothing)
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
      current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
        {
          'loss': loss.item(),
          'grad_norm': grad_norm.item(),
        }, global_step)
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
          global_step,
          loss.item(),
          grad_norm.item())

  return (optimizer_state, current_param_container, new_model_state)


def prepare_for_eval(workload: spec.Workload,
                     current_param_container: spec.ParameterContainer,
                     current_params_types: spec.ParameterTypeTree,
                     model_state: spec.ModelAuxiliaryState,
                     hyperparameters: spec.Hyperparameters,
                     loss_type: spec.LossType,
                     optimizer_state: spec.OptimizerState,
                     eval_results: List[Tuple[int, float]],
                     global_step: int,
                     rng: spec.RandomState) -> spec.UpdateReturn:
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
  if hasattr(HPARAMS, "batch_size"):
     return HPARAMS.batch_size
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


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
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
