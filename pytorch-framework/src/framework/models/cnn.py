import torch.nn as nn
from .base_model import BaseModel


class CNN(BaseModel):
    def __init__(self, input_channels, num_classes, channels=[32, 64, 128], kernel_size=3, 
                 activation='relu', dropout=0.0, pool_size=2):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        conv_layers = []
        prev_channels = input_channels
        
        for channel in channels:
            conv_layers.append(nn.Conv2d(prev_channels, channel, kernel_size, padding=1))
            if activation == 'relu':
                conv_layers.append(nn.ReLU())
            elif activation == 'tanh':
                conv_layers.append(nn.Tanh())
            conv_layers.append(nn.MaxPool2d(pool_size))
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            prev_channels = channel
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        self.fc_input_dim = channels[-1] * (28 // (pool_size ** len(channels))) ** 2
        self.fc_net = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return x