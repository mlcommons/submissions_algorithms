import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy

from muon_algos import MuonVanilla, MuonOG, MuonBucketed


# === Toy model (repeated 2D + 3D param shapes for bucket testing) ===
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Repeated 2D shapes
        self.fc1 = nn.Linear(8, 4, bias=False)  # (4×8)
        self.fc2 = nn.Linear(8, 4, bias=False)  # (4×8)
        self.fc3 = nn.Linear(3, 4, bias=False)  # (4×3)

        # --- Repeated 3D shapes
        self.conv1 = nn.Conv1d(4, 2, 3, padding=1, bias=False)  # (2×4×3)
        self.conv2 = nn.Conv1d(2, 2, 3, padding=1, bias=False)  # (2×2×3)
        self.conv3 = nn.Conv1d(2, 2, 3, padding=1, bias=False)  # (2×2×3)
        self.conv_unique = nn.Conv1d(2, 3, 2, padding=1, bias=False)  # (3×2×2)

    def forward(self, x_vec):
        # x_vec: (B, 8)
        y1 = self.fc1(x_vec)
        y2 = self.fc2(x_vec)
        y = (y1 + y2).relu()  # (B, 4)

        y = self.conv1(y.unsqueeze(2)).relu()  # (B, 2, 4)
        y = self.conv2(y).relu()  # (B, 2, 4)
        y = self.conv3(y).relu()  # (B, 2, 4)
        y = self.conv_unique(y).relu()  # (B, 3, 4)

        y = y.mean(dim=2)  # (B, 3)
        y = self.fc3(y)  # (B, 4)
        return y.mean()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # --- Init distributed (GPU, NCCL) ---
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    # --- Models ---
    model_v = ToyModel().to(device)
    model_og = deepcopy(model_v)
    model_b = deepcopy(model_v)

    # print(f"[RANK {rank}], {torch.__version__, dist.get_backend()}")
    # for p in model_v.parameters():
    #   print(f"RANK {rank}, p.shape: {p.shape}")
    # dist.barrier()

    ddp_v = DDP(model_v, device_ids=[rank])
    ddp_og = DDP(model_og, device_ids=[rank])
    ddp_b = DDP(model_b, device_ids=[rank])

    # --- Optimizers ---
    opt_v = MuonVanilla(ddp_v.parameters(), lr=0.01)
    opt_og = MuonOG(ddp_og.parameters(), lr=0.01)
    opt_b = MuonBucketed(ddp_b.parameters(), lr=0.01)

    # --- Multi-step training loop ---
    n_steps = 10
    for step in range(n_steps):
        x_vec = torch.randn(4, 8, device=device)
        for opt, model in [(opt_v, ddp_v), (opt_og, ddp_og), (opt_b, ddp_b)]:
            opt.zero_grad()
            loss = model(x_vec)
            loss.backward()
            opt.step()

        dist.barrier()
        if rank == 0:
            print(f"\n=== Step {step + 1}/{n_steps} Parameter Diffs ===")
        dist.barrier()
        for (n1, p1), (_, p2), (_, p3) in zip(
            ddp_v.module.named_parameters(),
            ddp_og.module.named_parameters(),
            ddp_b.module.named_parameters(),
        ):
            diff_v_og = (p1 - p2).abs().max().item()
            diff_v_b = (p1 - p3).abs().max().item()
            diff_og_b = (p2 - p3).abs().max().item()
            if rank == 0:
                print(
                    f"{n1:<20s} | V-OG Δ={diff_v_og:.2e} | V-B Δ={diff_v_b:.2e} | OG-B Δ={diff_og_b:.2e}"
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
