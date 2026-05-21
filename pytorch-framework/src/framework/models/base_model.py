import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, x):
        pass

    def compile(self, optimizer, loss_fn, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else []
        self.to(self.device)

    def train_step(self, x, y):
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), outputs

    def eval_step(self, x, y):
        self.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            outputs = self(x)
            loss = self.loss_fn(outputs, y)
        return loss.item(), outputs

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self(x)
            return outputs

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def summary(self, input_size):
        from torchsummary import summary
        summary(self, input_size)