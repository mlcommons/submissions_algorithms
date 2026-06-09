""""
Muon PyTorch DDP implementation.

The ``MuonDataParallel`` implementation reduce-scatters the gradients internally, distributing the parameter updates across gpus.

The current implementation flattens trailing dimension for any parameter with more than 2 dimensions.
While this is correct for 4D matrices, it might be problematic for some batched 3D parameters.
The current implementation is correct for AlgoPerf v07.0 workloads, but might require adjustments in future.
In its current stage ``MuonDataParallel`` does not support standard gradient clippin.
"""

import os
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
import logging


# Distributed settings
USE_DDP = 'RANK' in os.environ
RANK = int(os.environ['RANK']) if USE_DDP else 0
LOCAL_RANK = int(os.environ['LOCAL_RANK']) if USE_DDP else 0
WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if USE_DDP else 1

# Default values for Newton-Schulz
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_STEPS = 5
NS_EPS = 1e-7


@torch.compile()
def zeropower_via_newtonschulz5(G, steps=NS_STEPS, eps=NS_EPS):
  """
  Newton-Schulz iteration to approximally orthogonalize G.
  5-th order odd polynomial to approximate sign(x) on [-1,1],
  pushing singlular values to {+1,-1}.

  M = U @ S @ V.T
  sign(M) = U @ sign(S) @ V.T, odd matrix polynomial commutes with SVD
  sign(x) ~= a*x + b*x^3 + c*x^5, x in [-1,1]
  """
  if G.ndim != 2:
    raise RuntimeError(f"Expected 2D tensor in N-S, found {G.ndim} instead.")
  a, b, c = NS_A, NS_B, NS_C
  X = G.bfloat16()
  if G.size(0) > G.size(1):
    X = X.T

  # Ensure spectral norm is at most 1.
  # Ortho(cX)=Ortho(X), so we can normalize by ||X||_2 <= ||X||_F
  X /= (X.norm() + eps)

  # NS iterations
  for _ in range(steps):
    A = X @ X.T
    B = b * A + c * (A @ A)
    X = a * X + B @ X

  if G.size(0) > G.size(1):
    X = X.T
  return X


@torch.no_grad()
def muon_update(g, m, beta, nesterov, ns_steps, ns_eps):
  """Updates momentum ``m`` in-place and returns Muon update."""
  m.mul_(beta).add_(g, alpha=1 - beta)

  if nesterov:
    g = g.mul(1-beta).add(m, alpha=beta)
  else:
    g = m

  if g.ndim == 4:
    g = g.reshape(g.size(0), -1)  # flatten trailing dims on 4D params
  g = zeropower_via_newtonschulz5(g, steps=ns_steps, eps=ns_eps)
  if m.ndim == 4:
    g = g.view(m.shape)  # restore original shape

  return g
  

def _adjust_lr_to_match_adam(lr, param_shape):
  # https://arxiv.org/pdf/2502.16982
  A, B = param_shape[:2]
  return lr * 0.2 * (max(A, B) ** 0.5)


def _adjust_lr_spectral_norm(lr, param_shape):
  # Adjust from spectral norm 1 to RMS operator norm 1
  # https://arxiv.org/abs/2310.17813
  fan_out, fan_in = param_shape[:2]
  return lr * max(1.0, (fan_out / fan_in) ** 0.5)


def _param_to_complexity(p: torch.Tensor) -> int:
  """Compute NS complexity on p.grad."""
  # Shape after flatting potential trailing dims (3D, 4D)
  m, n = (p.shape[0], torch.tensor(p.shape[1:]).prod().item())
  # X @ X.T complexity: m^2n
  # XX.T @ XX.T complexity: m^3
  # XX.TXX.T @ X complexity: m^2n
  return 2 * (m ** 2) * n + m ** 3


class MuonBase(torch.optim.Optimizer, ABC):
  """
  Muon optimizer - Momentum Orthogonalized by Newton-Schulz.

  Abstract class.
  """
  
  def __init__(
    self,
    params,
    lr=0.02,
    weight_decay=0.0,
    beta=0.95,
    nesterov=True,
    ns_steps=NS_STEPS,
    ns_eps=NS_EPS,
    adjust_lr=None,
  ):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay: {weight_decay}')
    if not 0.0 <= beta < 1.0:
      raise ValueError(f'Invalid muon_beta parameter: {beta}')
    if nesterov not in [True, False]:
      raise ValueError(f'Invalid nesterov parameter: {nesterov}')
    if not 0 < ns_steps:
      raise ValueError(f'Invalid ns_steps parameter: {ns_steps}')
    if not 0.0 <= ns_eps:
      raise ValueError(f'Invalid ns_eps parameter: {ns_eps}')
    if not adjust_lr in [None, 'spectral_norm', 'match_adam']:
      raise ValueError(f'Invalid adjust_lr parameter: {adjust_lr}')

    defaults = dict(
      lr = lr,
      weight_decay = weight_decay,
      beta = beta,
      nesterov = nesterov,
      ns_steps = ns_steps,
      ns_eps = ns_eps,
    )
    super().__init__(params, defaults)

    if adjust_lr is None:
      self._adjust_lr = lambda lr, param_shape: lr
    elif adjust_lr == 'spectral_norm':
      self._adjust_lr = _adjust_lr_spectral_norm
    elif adjust_lr == 'match_adam':
      self._adjust_lr = _adjust_lr_to_match_adam

  @abstractmethod
  @torch.no_grad()
  def step(self, closure=None):
    pass


class MuonSingleDevice(MuonBase):
  """
  Single Devide implementation: if used with DDP, 
  it will replicate computation across devices.
  """
  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)


  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      wd = group['weight_decay']
      beta = group['beta']
      nesterov = group['nesterov']
      ns_steps = group['ns_steps']
      ns_eps = group['ns_eps']

      for p in group['params']:
        if p.grad is None:
          continue
        g = p.grad
        state = self.state[p]

        if len(state) == 0:
          state['m'] = torch.zeros_like(p)

        g = muon_update(g, state['m'], beta=beta, nesterov=nesterov, ns_steps=ns_steps, ns_eps=ns_eps)

        adjusted_lr = self._adjust_lr(lr, p.shape) # optionally adjust lr
        p.mul_(1 - lr * wd) # weigth decay
        p.add_(g, alpha=-adjusted_lr)

    return loss


def split_params_muon_adam(model):
  """Split parameters:
    - Muon: all matrix params (ndim ≥ 2) except embeddings
    - Adam: 1D params, all embeddings
  """
  muon_params, adam_params = [], []
  muon_infos, adam_infos = [], [] # for logging purposes

  for n, p in model.named_parameters():
    if not p.requires_grad:
      continue

    # Assign embeddings to Adam (wmt, criteo1tb, finewebedu_lm)
    if "embedding" in n.lower() or "embed_tokens" in n.lower():
      adam_params.append(p)
      adam_infos.append(f'{n} (ndim={p.ndim})')
    elif p.ndim >= 2:
      muon_params.append(p)
      muon_infos.append(f'{n} (ndim={p.ndim})')
    else:
      adam_params.append(p)
      adam_infos.append(f'{n} (ndim={p.ndim})')

  logging.info("Muon params:\n\t" + "\n\t".join(muon_infos))
  logging.info("Adam params:\n\t" + "\n\t".join(adam_infos))
  
  return muon_params, adam_params
