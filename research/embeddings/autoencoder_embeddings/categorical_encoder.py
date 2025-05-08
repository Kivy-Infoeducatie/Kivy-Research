import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    def __init__(self, num_tags=305, num_categories=300, embedding_dim=32):
        super(CategoricalEncoder, self).__init__()
        self.tag_embedding = nn.Embedding(
            num_embeddings=num_tags, embedding_dim=embedding_dim
        )
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

    def forward(self, tags, categories):
        # Encode tags and categories
        tags_encoded = torch.sum(
            self.tag_embedding(tags), dim=1
        )  # Sum embeddings for multi-label tags
        categories_encoded = torch.sum(
            self.category_embedding(categories), dim=1
        )  # Sum embeddings for multi-label categories
        return tags_encoded, categories_encoded
