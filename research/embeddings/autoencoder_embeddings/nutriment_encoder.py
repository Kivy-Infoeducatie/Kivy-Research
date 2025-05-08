import torch
import torch.nn as nn


class NutrimentEncoder(nn.Module):
    def __init__(self):
        super(NutrimentEncoder, self).__init__()
        self.nutriment_encoder = nn.Linear(
            9, 64
        )  # Encode 9 nutritional features into 64 dimensions

    def forward(self, nutriments):
        return self.nutriment_encoder(nutriments)
