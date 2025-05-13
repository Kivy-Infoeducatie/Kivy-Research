import torch
import torch.nn as nn
from torch import tensor


class CategoricalEncoder(nn.Module):
    def __init__(self, num_tags, num_categories, embedding_dim):
        super(CategoricalEncoder, self).__init__()
        self.tag_embedding = nn.Embedding(
            num_embeddings=num_tags, embedding_dim=embedding_dim
        )

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

    def forward(self, tags, categories):
        tags_encoded = torch.sum(self.tag_embedding(tags), dim=0)

        categories_encoded = self.category_embedding(categories)

        return tags_encoded, categories_encoded
