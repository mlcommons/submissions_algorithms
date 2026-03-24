# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, Iterator, List, Tuple
from absl import logging
import torch
import torch.distributed.nn as dist_nn
import torch.distributed as dist 
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]

class AdamWScheduleFreeV2(torch.optim.Optimizer):
    r"""Schedule Free AdamW
        This version differs from the earlier algoperf submitted ScheduleFree
        in the following ways:
         - It more closely follows the published version of the method, the 
            AlgoPerf submission was based on an early development version.
        - Weight decay is applied on the y sequence instead of the z sequence, 
          following the published version. This doesn't improve the results but
          is more for consistency.
        - A r=0.5 weighting sequence is no longer used. This simplifies the 
          implementation without any performance hit.

        The following change was also made to the outer loop:
        - Batchnorm buffers are now correctly updated, greatly improving the 
          performance on the ResNet and DeepSpeech workloads.
    """
    def __init__(self, params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 weight_lr_power=2,
                 warmup_steps=0,
                 ):
        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        k=0,
                        weight_sum=0.0,
                        warmup_steps=warmup_steps,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)

        super().__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Swap to extrapolated point:
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            k = group['k']

            for p in group['params']:
                # State initialization
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                z = state['z']

                # Extrapolate
                p.data.lerp_(end=z, weight=1-beta1)

        # Evaluate gradient at extrapolated point
        loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            k = group['k']
            warmup_steps = group['warmup_steps']
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            weight_lr_power = group['weight_lr_power']        

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            annealed_lr = group['lr']*sched

            lr = max(annealed_lr, eps)

            weight = lr**weight_lr_power
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            ckp1 = weight/weight_sum

            bias_correction2 = 1 - beta2 ** (k+1)
            step_size = lr * math.sqrt(bias_correction2)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']
                z = state['z']
                
                y = p.data.clone()

                # Unextrapolate (y -> x)
                p.data.lerp_(end=z, weight=1-1/beta1)

                # Decay at y (differs from V1)
                z.sub_(y, alpha=step_size*decay)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                # Apply bias correction to denominator (differs from V1)
                denom = exp_avg_sq.sqrt().add_(eps)

                # Take step on z
                z.addcdiv_(grad, denom, value=-step_size)

                ### Take step on x
                p.data.lerp_(end=z, weight=ckp1)

            group['k'] = k+1
        return loss


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state

  optimizer = AdamWScheduleFreeV2(
    model_params.parameters(),
    lr=hyperparameters.learning_rate,
    betas=(1.0 - hyperparameters.one_minus_beta1, hyperparameters.beta2),
    warmup_steps=int(hyperparameters.warmup_factor * workload.step_hint * 0.75),
    weight_decay=hyperparameters.weight_decay)

  optimizer_state = {'optimizer':optimizer}
  return optimizer_state

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type

  # Detect if BN
  current_model = current_param_container
  contains_bn = any(hasattr(m, "running_mean") for m in current_model.modules())

  if contains_bn and global_step % 3:
    # Update batch-norm statistics at the eval point x not y.
    with torch.no_grad():
      _, new_model_state = workload.model_fn(
              params=current_model,
              augmented_and_preprocessed_input_batch=batch,
              model_state=model_state,
              mode=spec.ForwardPassMode.TRAIN,
              rng=rng,
              update_batch_norm=True)
      model_state = new_model_state

  new_model_state = None

  def closure():
    nonlocal new_model_state
    optimizer_state['optimizer'].zero_grad()

    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=False)

    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits_batch,
        mask_batch=batch.get('weights'),
        label_smoothing=hyperparameters.label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    if USE_PYTORCH_DDP:
      # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
      summed_loss = dist_nn.all_reduce(summed_loss)
      n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

    loss.backward()
    return loss

  loss = optimizer_state['optimizer'].step(closure)

  return (optimizer_state, current_param_container, new_model_state)

def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 16
  elif workload_name == 'imagenet_resnet':
    return 768
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 128
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_resnet_silu':
    return 512
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