import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from src.config import Config
from src.models.base import BaseModel
from src.training.trainer import Trainer, launch_batch_training_task
from src.training.optimizer import OptimizerFactory
from src.training.scheduler import SchedulerFactory
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.logger import ModelLogger
from src.utils.seed import set_seed


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    set_seed(42)
    
    # Load data
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = MnistModel()
    optimizer = OptimizerFactory.create(model, 'adam', lr=0.001)
    scheduler = SchedulerFactory.create(optimizer, 'cosine', T_max=10)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('outputs/checkpoints', monitor='val_acc', mode='max', verbose=True),
        EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=True)
    ]
    
    # Train
    trainer = Trainer(model, optimizer, callbacks=callbacks, 
                     device='cuda' if torch.cuda.is_available() else 'cpu')
    
    history = trainer.train(train_loader, val_loader, epochs=10)
    print("Training completed!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
