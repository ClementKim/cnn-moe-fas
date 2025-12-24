import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, input_channels = 36, embed_dim = 128, num_experts = 4, top_k = 2):
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
            nn.Linear(feature_flat_dim, num_experts)
        )

        self.experts = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(feature_flat_dim, 256),
                  nn.ReLU(),
                  nn.Dropout(0.2),
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

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s = 30.0, m = 0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device = x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output