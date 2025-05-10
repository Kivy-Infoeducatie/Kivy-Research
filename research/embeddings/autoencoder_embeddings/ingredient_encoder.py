import torch
import torch.nn as nn
from torch import tensor


class IngredientEncoder(nn.Module):
    def __init__(self, ingredient_vocab_size, embedding_dim):
        super(IngredientEncoder, self).__init__()
        self.ingredient_embedding = nn.Embedding(
            num_embeddings=ingredient_vocab_size, embedding_dim=embedding_dim
        )

    def forward(self, ingredient_ids):
        ingredient_embeds = torch.sum(
            self.ingredient_embedding(tensor(ingredient_ids, dtype=torch.int64)), dim=1
        )

        return ingredient_embeds
