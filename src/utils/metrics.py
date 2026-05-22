import torch
from typing import Dict


class Metrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
    
    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor):
        self.loss_sum += loss * targets.size(0)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        return {
            'accuracy': self.correct / self.total if self.total > 0 else 0.0,
            'loss': self.loss_sum / self.total if self.total > 0 else 0.0
        }
