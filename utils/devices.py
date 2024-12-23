import torch


def cpu() -> torch.device:
    """Get the CPU device."""
    return torch.device("cpu")


def gpu(i: int = 0) -> torch.device:
    """Get a GPU device."""
    return torch.device(f"cuda:{i}")


def num_gpus() -> torch.device:
    """Get the number of available GPUs."""
    return torch.cuda.device_count()


def try_gpu(i: int = 0) -> torch.device | None:
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return None


def try_all_gpus() -> torch.device:
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]
