import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, num_classes = 10, num_experts = 4, top_k = 2):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.router = nn.Sequential(
            nn.Linear(7 * 7 * 32, num_experts),
            nn.Softmax(dim = 1)
        )

        self.expert = nn.Sequential(
            nn.Linear(7 * 7 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        self.experts_list = []
        for _ in range(num_experts):
            self.experts_list.append(self.expert)

        self.experts = nn.ModuleList(self.experts_list)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)

        routing_weights = self.router(out)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        all_expert_outputs = torch.stack([expert(out) for expert in self.experts], dim=1) # (batch_size, num_experts, num_classes)
        top_k_expert_outputs = torch.gather(all_expert_outputs, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.num_classes))
        weighted_outputs = top_k_expert_outputs * top_k_weights.unsqueeze(-1)

        final_output = torch.sum(weighted_outputs, dim=1)

        return final_output