""""
Muon torch implementations.
"""

import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from algoperf.pytorch_utils import pytorch_setup


USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
WORLD_SIZE = N_GPUS # single-node assumption


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
  if G.ndim != 2:
    raise RuntimeError(f"Expected 2D tensor in N-S, found {G.ndim} instead.")
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


class MuonBase(torch.optim.Optimizer, ABC):
  """Muon optimizer - Momentum Orthogonalized by Newton-Schulz.

  Abstract class.
  """
  
  def __init__(
    self,
    params,
    lr=0.02,
    weight_decay=0.0,
    beta=0.95,
    nesterov=True,
    ns_steps=5,
    ns_eps=1.0e-7,
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

    defaults = dict(
      lr = lr,
      weight_decay = weight_decay,
      beta = beta,
      nesterov = nesterov,
      ns_steps = ns_steps,
      ns_eps = ns_eps,
    )
    super().__init__(params, defaults)

  @abstractmethod
  @torch.no_grad()
  def step(self, closure=None):
    pass


class MuonVanilla(MuonBase):
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
        state['m'].mul_(beta).add_(g, alpha=1 - beta)

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

    return loss


class MuonKJ(MuonBase):
  """
  From: https://github.com/KellerJordan/Muon/blob/master/muon.py#L98

  For each parameter group, (sorted) parameters are processed in blocks of world_size. 
  Each block is distributed round-robin across ranks; 
  each device updates its assigned parameters locally, 
  then all_gather syncs the block across ranks.
  
  Comms: one all-gather per block -> ~#params/WORLD_SIZE comms.
  Space: O(largest_param);
         O(WORLD_SIZE * largest_param) transient during all_gather.
  """
  def __init__(self, params, **kwargs):
    if not isinstance(params, list):
        params = list(params) 
    # Sort params by shape, to fairly distribute orthogonalization across devices
    if isinstance(params[0], dict): # sort each param group individually
      for group in params:
        group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
    else:
      params = sorted(params, key=lambda x: x.size(), reverse=True)

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
      params = group['params']

      # Pad params so each all-gather block is of size WORLD_SIZE.
      # list concat keeps param refs (not copies), so all_gather updates model params directly.      
      pad = (WORLD_SIZE - len(params) % WORLD_SIZE) % WORLD_SIZE
      params_pad = params + [torch.empty_like(params[-1])] * pad

      # Iterate over params in blocks of WORLD_SIZE
      for block_start in range(0, len(params), WORLD_SIZE):
        # Each device updates the RANK-th tensor in the block
        if block_start + RANK < len(params): # skip padded tensors
          p = params[block_start + RANK] # round-robin
          if p.grad is None:
            p.grad = torch.zeros_like(p) # ensure valid tensor for all_gather
          g = p.grad
          state = self.state[p]

          if len(state) == 0:
            state['m'] = torch.zeros_like(p)
          state['m'].mul_(beta).add_(g, alpha=1 - beta)

          if nesterov:
            g = g.add(state['m'], alpha=beta)
          else:
            g = state['m']
          g = g.reshape(g.size(0), -1)  # flatten trailing dims on 3D, 4D params
          g = zeropower_via_newtonschulz5(g, steps=ns_steps, eps=ns_eps)
          g = g.view(p.shape)  # restore original shape on 3D, 4D params
          g *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5 # Adjust from spectral norm 1 to RMS operator norm 1 https://arxiv.org/abs/2310.17813

          p.mul_(1 - lr * wd)
          p.add_(g, alpha=-lr)
        
        # all-gather current block of params (including padded entries)
        dist.all_gather(params_pad[block_start:block_start + WORLD_SIZE], 
                        params_pad[block_start + RANK])

    return loss


class MuonBucketed(MuonBase):
  """
  Muon with shape-based bucketing.

  Each param group is split into buckets of same-shaped params.  
  Within each bucket, parameters are processed in WORLD_SIZE-sized blocks:  
  each block is distributed round-robin across ranks, each rank updates its share,  
  then an all_gather synchronizes the updated block across devices.
  
  Compared to `MuonKJ`, bucketing adds only an outer loop (grouped by shape),
  but does not increase total all-gather calls — it just organizes them by shape. 
  Communication volume and count remain essentially unchanged.
  
  Comms: one all-gather per block -> ~#params/WORLD_SIZE calls.
  Space: O(largest_param) persistent;
         O(WORLD_SIZE * largest_param) transient during all_gather.
  """
  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    for group in self.param_groups:
      # Pre-bucket params by shape; bucket is a list of params
      group['bucket'] = {}
      for p in group['params']:
        group['bucket'].setdefault(p.shape, []).append(p)

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
      
      for shape, params in group['bucket'].items():
        # Pad params so each all-gather block is of size WORLD_SIZE.
        # list concat keeps param refs (not copies), so all_gather updates model params directly.
        pad = (WORLD_SIZE - len(params) % WORLD_SIZE) % WORLD_SIZE
        params_pad = params + [torch.empty_like(params[-1])] * pad
        
        # Iterate over params in blocks of WORLD_SIZE
        for block_start in range(0, len(params), WORLD_SIZE):
          if block_start + RANK < len(params): # skip padded tensors
            p = params[block_start + RANK] # round-robin
            if p.grad is None:
              p.grad = torch.zeros_like(p) # ensure valid tensor for all_gather
            g = p.grad
            state = self.state[p]

            if len(state) == 0:
              state['m'] = torch.zeros_like(p)
            state['m'].mul_(beta).add_(g, alpha=1 - beta)

            if nesterov:
              g = g.add(state['m'], alpha=beta)
            else:
              g = state['m']
            g = g.reshape(g.size(0), -1)  # flatten trailing dims on 3D, 4D params
            g = zeropower_via_newtonschulz5(g, steps=ns_steps, eps=ns_eps)
            g = g.view(p.shape)  # restore original shape on 3D, 4D params
            g *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5 # Adjust from spectral norm 1 to RMS operator norm 1 https://arxiv.org/abs/2310.17813

            p.mul_(1 - lr * wd)
            p.add_(g, alpha=-lr)
          
          # all-gather current block of params (including padded entries)
          dist.all_gather(params_pad[block_start:block_start + WORLD_SIZE],
                          params_pad[block_start + RANK])

    return loss 
          
