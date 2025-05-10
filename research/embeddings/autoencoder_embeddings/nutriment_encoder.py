import torch
import torch.nn as nn


class NutrimentEncoder(nn.Module):
    def __init__(self, nutriment_num, embedding_dim):
        super(NutrimentEncoder, self).__init__()
        self.nutriment_encoder = nn.Linear(nutriment_num, embedding_dim)

    def forward(self, nutriments):
        return self.nutriment_encoder(nutriments)
