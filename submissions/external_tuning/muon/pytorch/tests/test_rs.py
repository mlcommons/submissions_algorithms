import torch, torch.distributed as dist, torch.multiprocessing as mp


def run(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    t1 = torch.randn(2, 3, device=f"cuda:{rank}")
    t2 = torch.randn(4, 5, device=f"cuda:{rank}")  # different shape
    grad_block = [t1, t2]
    out = torch.zeros_like(t1)

    try:
        dist.reduce_scatter(out, grad_block)
        print(f"Rank {rank}: success")
    except Exception as e:
        print(f"Rank {rank} error:", e)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
