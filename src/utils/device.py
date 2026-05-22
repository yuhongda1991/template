import torch
from typing import Optional


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = get_device()
    return torch.device(device)


def to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    return data
