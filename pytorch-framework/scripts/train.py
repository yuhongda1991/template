import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.framework.config import Config
from src.framework.models import CNN, MLP
from src.framework.data import DataLoader, Dataset
from src.framework.trainer import Trainer


def main():
    config = Config({
        'batch_size': 64,
        'epochs': 10,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_fn': 'cross_entropy',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001
    })

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CNN(input_channels=1, num_classes=10, channels=[32, 64, 128], dropout=0.2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    trainer = Trainer(model, config)
    
    print(f"Training on device: {model.device}")
    trainer.train(train_loader, test_loader, epochs=config['epochs'])
    
    print("\nEvaluating on test set:")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")


if __name__ == '__main__':
    main()