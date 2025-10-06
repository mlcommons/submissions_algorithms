import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy

from muon_algos import MuonBucketed


# === Toy model (repeated 2D + 3D param shapes for bucket testing) ===
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.fc3 = nn.Linear(3, 4, bias=False)

        self.conv1 = nn.Conv1d(4, 2, 3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(2, 2, 3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(2, 2, 3, padding=1, bias=False)
        self.conv_unique = nn.Conv1d(2, 3, 2, padding=1, bias=False)

    def forward(self, x_vec):
        y1 = self.fc1(x_vec)
        y2 = self.fc2(x_vec)
        y = (y1 + y2).relu()

        y = self.conv1(y.unsqueeze(2)).relu()
        y = self.conv2(y).relu()
        y = self.conv3(y).relu()
        y = self.conv_unique(y).relu()

        y = y.mean(dim=2)
        y = self.fc3(y)
        return y.mean()



def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    # --- Two identical models ---
    model_a = ToyModel().to(device)
    model_b = deepcopy(model_a)

    ddp_a = DDP(model_a, device_ids=[rank])
    ddp_b = DDP(model_b, device_ids=[rank])

    opt_a = MuonBucketed(ddp_a.parameters(), lr=0.01)
    opt_b = MuonBucketed(ddp_b.parameters(), lr=0.01)

    n_steps = 10
    for step in range(n_steps):
        x_vec = torch.randn(4, 8, device=device)

        # --- Normal DDP backward ---
        opt_a.zero_grad()
        loss_a = ddp_a(x_vec)
        loss_a.backward()
        opt_a.step()

        # --- Manual reduce-scatter backward ---
        opt_b.zero_grad()
        loss_b = ddp_b(x_vec)

        ddp_b.require_backward_grad_sync = False  # disable default sync
        loss_b.backward()

        # Manually average grads via reduce_scatter
        for group in opt_b.param_groups:
            for shape, params in group["bucket"].items():
                for block_start in range(0, len(params), world_size):
                    grad_block = [
                        (p.grad if p.grad is not None else torch.zeros_like(p))
                        for p in params[block_start:block_start + world_size]
                    ]
                    pad = world_size - len(grad_block)
                    if pad > 0:
                        grad_block += [torch.zeros_like(grad_block[-1])] * pad

                    grad_shard = torch.zeros_like(grad_block[rank])
                    dist.reduce_scatter(grad_shard, grad_block)
                    grad_shard.div_(world_size)

                    if block_start + rank < len(params):
                        params[block_start + rank].grad = grad_shard

        opt_b.step()

        # --- Compare ---
        dist.barrier()
        if rank == 0:
            print(f"\n=== Step {step+1}/{n_steps} Reduce-Scatter Check ===")
        dist.barrier()

        for (n1, p1), (n2, p2) in zip(
            ddp_a.module.named_parameters(),
            ddp_b.module.named_parameters(),
        ):
            diff = (p1 - p2).abs().max().item()
            if rank == 0:
                print(f"{n1:<20s} | Δ={diff:.2e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
