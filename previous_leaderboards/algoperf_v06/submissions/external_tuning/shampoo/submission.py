"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Submission file for Distributed Shampoo optimizer with possible linear warmup + cosine decay LR scheduler in PyTorch."""

from typing import Dict, Iterator, List, Optional, Tuple

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup
# assume submissions_algorithms will be mounted at top level dir of algorithmic-efficiency repo
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    AdaGradGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    GraftingConfig,
    RMSpropGraftingConfig,
    RWSAdaGradGraftingConfig,
    SGDGraftingConfig,
)

USE_PYTORCH_DDP = pytorch_setup()[0]


def init_optimizer_state(
        workload: spec.Workload,
        model_params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        rng: spec.RandomState
) -> spec.OptimizerState:
    """Creates Distributed Shampoo optimizer and learning rate scheduler."""
    del model_state
    del rng

    str_to_communication_dtype = {
        "FP32": CommunicationDType.FP32,
        "FP16": CommunicationDType.FP16,
        "BF16": CommunicationDType.BF16,
    }

    # Instantiate grafting configuration.
    grafting_config = instantiate_grafting_config(
        grafting_type=hyperparameters.grafting_type,
        grafting_beta2=1.0 - hyperparameters.one_minus_beta2,
        grafting_epsilon=hyperparameters.grafting_epsilon,
    )

    shampoo_params = []
    rws_adagrad_params = []
    for param in model_params.parameters():
        if torch.any(torch.tensor(param.shape) >= 1_000_000):
            rws_adagrad_params.append(param)
        else:
            shampoo_params.append(param)

    optimizer_state = {
        'optimizer': DistributedShampoo(
            shampoo_params,
            lr=hyperparameters.learning_rate,
            betas=(
                1.0 - hyperparameters.one_minus_beta1,
                1.0 - hyperparameters.one_minus_beta2
            ),
            epsilon=hyperparameters.epsilon,
            momentum=1.0 - hyperparameters.one_minus_momentum if hyperparameters.use_momentum else 0.0,
            weight_decay=hyperparameters.weight_decay,
            max_preconditioner_dim=hyperparameters.max_preconditioner_dim,
            precondition_frequency=hyperparameters.precondition_frequency,
            start_preconditioning_step=hyperparameters.start_preconditioning_step,
            inv_root_override=hyperparameters.inv_root_override,
            exponent_multiplier=hyperparameters.exponent_multiplier,
            use_nadam=hyperparameters.use_nadam,
            use_nesterov=True,
            use_bias_correction=True,
            use_decoupled_weight_decay=True,
            grafting_config=grafting_config,
            use_normalized_grafting=hyperparameters.use_normalized_grafting,
            use_merge_dims=True,
            use_pytorch_compile=False,
            distributed_config=DDPShampooConfig(
                communication_dtype=str_to_communication_dtype[hyperparameters.communication_dtype],
                num_trainers_per_group=8,
                communicate_params=hyperparameters.communicate_params,
            ),
            preconditioner_dtype=torch.float32,
            use_protected_eigh=True,
            track_root_inv_residuals=False,
        ),
    }

    if len(rws_adagrad_params) > 0:
        logging.info("Large parameters (embedding tables) detected (dim >= 1_000_000). Instantiating Row-Wise AdaGrad...")
        optimizer_state["rws_optimizer"] = DistributedShampoo(
            rws_adagrad_params,
            lr=hyperparameters.learning_rate,
            betas=(
                1.0 - hyperparameters.one_minus_beta1,
                1.0 - hyperparameters.one_minus_beta2
            ),
            epsilon=hyperparameters.epsilon,
            momentum=1.0 - hyperparameters.one_minus_momentum if hyperparameters.use_momentum else 0.0,
            weight_decay=hyperparameters.weight_decay,
            max_preconditioner_dim=524_288,
            precondition_frequency=hyperparameters.precondition_frequency,
            start_preconditioning_step=torch.inf,
            inv_root_override=hyperparameters.inv_root_override,
            exponent_multiplier=hyperparameters.exponent_multiplier,
            use_nadam=hyperparameters.use_nadam,
            use_nesterov=True,
            use_bias_correction=True,
            use_decoupled_weight_decay=True,
            grafting_config=AdamGraftingConfig(
                beta2=1.0 - hyperparameters.one_minus_beta2,
                epsilon=hyperparameters.grafting_epsilon,
            ),
            use_normalized_grafting=hyperparameters.use_normalized_grafting,
            use_merge_dims=True,
            use_pytorch_compile=False,
            distributed_config=DDPShampooConfig(
                communication_dtype=str_to_communication_dtype[hyperparameters.communication_dtype],
                num_trainers_per_group=8,
                communicate_params=hyperparameters.communicate_params,
            ),
            preconditioner_dtype=torch.float32,
            use_protected_eigh=True,
            track_root_inv_residuals=False,
        )
    else:
        logging.info("No large parameters detected! Continuing with only Shampoo....")

    def linear_warmup_and_cosine_decay(
        step_hint: int,
        hyperparameters,
        optimizer
    ):
        warmup_steps = int(hyperparameters.warmup_factor * step_hint)
        warmup = LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.,
            total_iters=warmup_steps,
        )
        cosine_steps = max(step_hint - warmup_steps, 1)
        cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine_decay] if hyperparameters.use_cosine_decay else [warmup],
            milestones=[warmup_steps] if hyperparameters.use_cosine_decay else []
        )

    optimizer_state['scheduler'] = linear_warmup_and_cosine_decay(
        hyperparameters.step_hint_factor * workload.step_hint,
        hyperparameters,
        optimizer_state['optimizer']
    )
    if 'rws_optimizer' in optimizer_state:
        optimizer_state['rws_scheduler'] = linear_warmup_and_cosine_decay(
            hyperparameters.step_hint_factor * workload.step_hint,
            hyperparameters,
            optimizer_state['rws_optimizer']
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
    rng: spec.RandomState
) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types
    del loss_type
    del eval_results

    current_model = current_param_container
    current_model.train()
    optimizer_state['optimizer'].zero_grad()
    if 'rws_optimizer' in optimizer_state:
        optimizer_state['rws_optimizer'].zero_grad()

    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True
    )

    label_smoothing = hyperparameters.label_smoothing

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
        summed_loss = dist_nn.all_reduce(summed_loss)
        n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            current_model.parameters(),
            max_norm=grad_clip,
        )

    optimizer_state['optimizer'].step()
    if 'rws_optimizer' in optimizer_state:
        optimizer_state['rws_optimizer'].step()
    if "scheduler" in optimizer_state:
        optimizer_state['scheduler'].step()
    if 'rws_scheduler' in optimizer_state:
        optimizer_state['rws_scheduler'].step()

    # Log training metrics - loss, grad_norm, batch_size.
    if global_step <= 100 or global_step % 500 == 0:
        with torch.no_grad():
            parameters = [p for p in current_model.parameters() if p.grad is not None]
        grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
            2,
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


def instantiate_grafting_config(
    grafting_type: str,
    grafting_beta2: float,
    grafting_epsilon: float,
) -> Optional[GraftingConfig]:
    if grafting_type == "NONE":
        return None
    elif grafting_type == "ADAGRAD":
        return AdaGradGraftingConfig(
            epsilon=grafting_epsilon,
        )
    elif grafting_type == "ADAM":
        return AdamGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    elif grafting_type == "RMSPROP":
        return RMSpropGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    elif grafting_type == "SGD":
        return SGDGraftingConfig()
    else:
        raise ValueError(f"Invalid GraftingType {grafting_type}!")

def get_batch_size(workload_name: str) -> int:
    # Return the global batch size.
    if workload_name == 'criteo1tb':
        return 262_144
        # return 524_288
    elif workload_name == 'fastmri':
        return 32
        # return 64
    elif workload_name == 'imagenet_resnet':
        return 1024
    elif workload_name == 'imagenet_resnet_silu':
        return 512
    elif workload_name == 'imagenet_resnet_gelu':
        return 512
    elif workload_name == 'imagenet_vit':
        return 1024
    elif workload_name == 'librispeech_conformer':
        return 200
    elif workload_name == 'librispeech_deepspeech':
        return 224
    elif workload_name == 'ogbg':
        return 512
        #  return 1024
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
    rng: spec.RandomState
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
