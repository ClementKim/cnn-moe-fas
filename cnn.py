import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, input_channels = 36, embed_dim = 128):
        super(Backbone, self).__init__()

        feature_flat_dim = 7 * 7 * 32
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.linear = nn.Sequential(
            nn.Linear(feature_flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.embed_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # flatten

        x = self.linear(x)
        
        return x