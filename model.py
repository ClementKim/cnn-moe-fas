import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(torch.nn.Module):
    def __init__(self, input_channels = 3, embed_dim = 128, num_experts = 4, top_k = 2):
        super(Backbone, self).__init__()

        feature_flat_dim = 7 * 7 * 32
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

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

        self.router = nn.Sequential(
            nn.Linear(feature_flat_dim, num_experts),
            nn.Softmax(dim = 1)
        )

        self.experts = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(feature_flat_dim, 256),
                  nn.ReLU(),
                  nn.Linear(256, self.embed_dim)
              )
              for _ in range(num_experts)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # flatten

        routing_logits = self.router[0](x) # before softmax

        routing_weights = F.softmax(routing_logits, dim = 1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim = 1)

        top_k_weights = top_k_weights / top_k_weights.sum(dim = 1, keepdim = True)

        all_experts_outputs = torch.stack([expert(x) for expert in self.experts], dim = 1)

        batch_size = x.size(0)
        expanded_indices = top_k_indices.unsqueeze(-1).expand(batch_size, self.top_k, self.embed_dim)

        selected_experts_outputs = torch.gather(all_experts_outputs, 1, expanded_indices)

        weighted_outputs = selected_experts_outputs * top_k_weights.unsqueeze(-1)

        final_embedding = F.normalize(torch.sum(weighted_outputs, dim = 1), p = 2, dim = 1)

        return final_embedding

class ClassificationHead(torch.nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
class IdentificationHead(torch.nn.Module):
    def __init__(self, embed_dim, id_classes):
        super(IdentificationHead, self).__init__()
        self.fc = nn.Linear(embed_dim, id_classes)

    def forward(self, x):
        return self.fc(x)
    
class VerificationHead(torch.nn.Module):
    def __init__(self, embed_dim, ver_classes):
        super(VerificationHead, self).__init__()
        self.fc = nn.Linear(embed_dim, ver_classes)

    def forward(self, x):
        return self.fc(x)