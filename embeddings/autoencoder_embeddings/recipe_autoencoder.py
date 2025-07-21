import torch.nn as nn

from embeddings.autoencoder_embeddings.recipe_autoencoder_config import (
    RecipeAutoencoderConfig,
)
from embeddings.autoencoder_embeddings.recipe_decoder import RecipeDecoder
from embeddings.autoencoder_embeddings.recipe_encoder import RecipeEncoder


class RecipeAutoencoder(nn.Module):
    def __init__(
        self,
        ingredient_vocab_size,
        category_vocab_size,
        tag_vocab_size,
        num_nutriments,
    ):
        super(RecipeAutoencoder, self).__init__()

        config = RecipeAutoencoderConfig(
            ingredient_vocab_size=ingredient_vocab_size,
            category_vocab_size=category_vocab_size,
            tag_vocab_size=tag_vocab_size,
            num_nutriments=num_nutriments,
        )

        self.encoder = RecipeEncoder(config)
        self.decoder = RecipeDecoder(config)

    def forward(
        self, name, steps, tags, categories, ingredient_ids, nutriments
    ):
        latent_layer = self.encoder(
            name, steps, tags, categories, ingredient_ids, nutriments
        )

        return self.decoder(latent_layer)
