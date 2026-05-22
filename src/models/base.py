import torch
from torch import nn
from typing import Optional, Dict, Any


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def summary(self) -> Dict[str, Any]:
        return {
            'num_params': self.get_num_params(),
            'trainable_params': self.get_trainable_params(),
            'layers': list(self.named_children())
        }
