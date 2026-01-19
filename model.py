# Import necessary libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone network for feature extraction
class Backbone(nn.Module):
    """
    The main feature extraction network, which uses a Mixture of Experts (MoE) model.
    This network processes an input image and outputs a feature embedding.
    """
    def __init__(self, input_channels = 36, embed_dim = 256, num_experts = 8, top_k = 2):
        """
        Initializes the Backbone network.

        Args:
            input_channels (int): Number of input channels for the first convolutional layer.
            embed_dim (int): Dimension of the output feature embedding.
            num_experts (int): The total number of expert networks in the MoE layer.
            top_k (int): The number of experts to use for each input.
        """
        super(Backbone, self).__init__()

        # The dimension of the flattened feature map after pooling.
        feature_flat_dim = 7 * 7 * 128
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Convolutional layers for initial feature extraction.
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

        # Adaptive average pooling to ensure a fixed-size feature map.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # The router network that determines which experts to use for a given input.
        self.router = nn.Sequential(
            nn.Linear(feature_flat_dim, num_experts)
        )

        # A list of expert networks. Each expert is a small feed-forward network.
        self.experts = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(feature_flat_dim, 512),
                  nn.ReLU(),
                  nn.Dropout(0.2),
                  nn.Linear(512, self.embed_dim)
              )
              for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Defines the forward pass of the Backbone network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The final feature embedding.
        """
        # Extract and flatten features from the input.
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 

        # Get routing weights from the router and apply softmax.
        routing_logits = self.router[0](x)
        routing_weights = F.softmax(routing_logits, dim = 1)

        # Select the top-k experts based on the routing weights.
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim = 1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim = 1, keepdim = True)

        # Get outputs from all experts.
        all_experts_outputs = torch.stack([expert(x) for expert in self.experts], dim = 1)

        # Gather the outputs of the selected top-k experts.
        batch_size = x.size(0)
        expanded_indices = top_k_indices.unsqueeze(-1).expand(batch_size, self.top_k, self.embed_dim)
        selected_experts_outputs = torch.gather(all_experts_outputs, 1, expanded_indices)

        # Weight the expert outputs and sum them to get the final embedding.
        weighted_outputs = selected_experts_outputs * top_k_weights.unsqueeze(-1)
        final_embedding = F.normalize(torch.sum(weighted_outputs, dim = 1), p = 2, dim = 1)

        return final_embedding

# Classification head for the model
class ClassificationHead(nn.Module):
    """
    A simple classification head with a single linear layer.
    This is used to classify the features extracted by the backbone.
    """
    def __init__(self, input_dim, num_classes):
        """
        Initializes the ClassificationHead.

        Args:
            input_dim (int): The dimension of the input features.
            num_classes (int): The number of output classes.
        """
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the classification head.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        return self.fc(x)

# ArcFace head for metric learning
class ArcFaceHead(nn.Module):
    """
    Implements the ArcFace loss, which is used for face recognition and other
    metric learning tasks. It encourages larger angular margins between classes.
    """
    def __init__(self, in_features, out_features, s = 30.0, m = 0.50):
        """
        Initializes the ArcFaceHead.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output classes (e.g., identities).
            s (float): The radius of the hypersphere on which the features are projected.
            m (float): The angular margin penalty.
        """
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # The weight parameter for the linear layer, which represents the class centers.
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cosine and sine of the margin for efficiency.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Thresholds used in the margin penalty.
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        """
        Defines the forward pass of the ArcFace head.

        Args:
            x (torch.Tensor): The input feature tensor.
            label (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The output logits.
        """
        # Compute the cosine similarity between the input features and the class centers.
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute the sine of the angle and the angle phi.
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply the margin penalty to the target class.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create a one-hot encoding of the labels.
        one_hot = torch.zeros(cosine.size(), device = x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combine the modified cosine (for the target class) and the original cosine.
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output