# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone network for feature extraction
class Backbone(nn.Module):
    """
    A convolutional neural network (CNN) backbone for feature extraction.
    This network takes an image with a specific number of channels and
    outputs a feature vector of a specified embedding dimension.
    """
    def __init__(self, input_channels = 36, embed_dim = 256):
        """
        Initializes the Backbone network.

        Args:
            input_channels (int): The number of input channels for the first convolutional layer.
            embed_dim (int): The dimension of the output feature vector (embedding).
        """
        super(Backbone, self).__init__()

        # The dimension of the flattened feature map after pooling.
        # This is calculated as 7 (height) * 7 (width) * 128 (channels).
        feature_flat_dim = 7 * 7 * 128
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        # Convolutional layers for extracting features from the input image.
        # This sequence of layers progressively reduces the spatial dimensions
        # while increasing the number of channels.
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Adaptive average pooling to resize the feature maps to a fixed size (7x7).
        # This allows the network to handle inputs of varying sizes.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers to produce the final embedding.
        # These layers take the flattened feature map and project it into the embedding space.
        self.linear = nn.Sequential(
            nn.Linear(feature_flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.embed_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the Backbone network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: The output embedding of shape (batch_size, embed_dim).
        """
        # Pass the input through the convolutional layers.
        x = self.features(x)
        # Apply adaptive pooling to standardize the feature map size.
        x = self.adaptive_pool(x)
        # Flatten the feature map for the linear layers.
        x = x.view(x.size(0), -1)
        # Pass the flattened features through the linear layers to get the embedding.
        x = self.linear(x)
        
        return x