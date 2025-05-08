import torch
import torch.nn as nn


class IngredientEncoder(nn.Module):
    def __init__(self, ingredient_vocab_size=10000, embedding_dim=64):
        super(IngredientEncoder, self).__init__()
        self.ingredient_embedding = nn.Embedding(
            num_embeddings=ingredient_vocab_size, embedding_dim=embedding_dim
        )

    def forward(self, ingredient_ids):
        # Encode ingredients using an embedding layer
        ingredient_embeds = torch.sum(
            self.ingredient_embedding(ingredient_ids), dim=1
        )  # Sum embeddings
        return ingredient_embeds
