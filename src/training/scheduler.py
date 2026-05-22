import torch
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
    ExponentialLR, MultiStepLR
)


class SchedulerFactory:
    _schedulers = {
        'step': StepLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'exponential': ExponentialLR,
        'multistep': MultiStepLR,
    }
    
    @classmethod
    def create(cls, optimizer, name: str = 'cosine', **kwargs):
        if name == 'plateau':
            return cls._schedulers[name](optimizer, **kwargs)
        return cls._schedulers[name](optimizer, **kwargs)
    
    @classmethod
    def register(cls, name: str, scheduler_cls):
        cls._schedulers[name] = scheduler_cls
