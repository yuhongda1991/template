import torch
from torch import nn


class OptimizerFactory:
    _optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop,
        'adagrad': torch.optim.Adagrad,
    }
    
    @classmethod
    def create(cls, model: nn.Module, name: str = 'sgd', **kwargs) -> torch.optim.Optimizer:
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(cls._optimizers.keys())}")
        
        optimizer_cls = cls._optimizers[name]
        return optimizer_cls(model.parameters(), **kwargs)
    
    @classmethod
    def register(cls, name: str, optimizer_cls):
        cls._optimizers[name] = optimizer_cls
