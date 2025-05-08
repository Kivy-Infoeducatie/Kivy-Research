import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(FusionLayer, self).__init__()
        self.fusion = nn.Linear(
            input_dim, latent_dim
        )  # Fuse features into latent space

    def forward(self, *features):
        # Concatenate all features and encode into latent space
        combined_features = torch.cat(features, dim=-1)
        return self.fusion(combined_features)
