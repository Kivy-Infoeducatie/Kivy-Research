import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FusionLayer, self).__init__()
        self.fusion = nn.Linear(input_dim, latent_dim)

    def forward(self, *features):
        combined_features = torch.cat(features, dim=-1)
        return self.fusion(combined_features)
