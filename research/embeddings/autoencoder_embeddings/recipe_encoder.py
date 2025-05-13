import torch
from torch import nn

from embeddings.autoencoder_embeddings.categorical_encoder import CategoricalEncoder
from embeddings.autoencoder_embeddings.fusion_layer import FusionLayer
from embeddings.autoencoder_embeddings.ingredient_encoder import IngredientEncoder
from embeddings.autoencoder_embeddings.nutriment_encoder import NutrimentEncoder
from embeddings.autoencoder_embeddings.recipe_autoencoder import RecipeAutoencoderConfig
from embeddings.autoencoder_embeddings.text_encoder import TextEncoder


class RecipeEncoder(nn.Module):
    def __init__(self, config: RecipeAutoencoderConfig):
        super(RecipeEncoder, self).__init__()

        self.text_encoder = TextEncoder(
            transformer_model=config.text_transformer_model,
            transformer_model_embedding_dim=config.text_embedding_dim,
            text_embedding_dim=config.text_latent_dim,
        )

        self.categorical_encoder = CategoricalEncoder(
            num_tags=config.tag_vocab_size,
            num_categories=config.category_vocab_size,
            embedding_dim=config.categorical_embedding_dim,
        )

        self.ingredient_encoder = IngredientEncoder(
            ingredient_vocab_size=config.ingredient_vocab_size,
            embedding_dim=config.ingredient_embedding_dim,
        )

        self.nutriment_encoder = NutrimentEncoder(
            nutriment_num=config.num_nutriments,
            embedding_dim=config.nutriment_embedding_dim,
        )

        self.fusion_layer = FusionLayer(
            input_dim=config.text_latent_dim  # Name
            + config.text_latent_dim  # Steps
            + config.categorical_embedding_dim  # Tags
            + config.categorical_embedding_dim  # Category
            + config.ingredient_embedding_dim  # Ingredients
            + config.nutriment_embedding_dim,  # Nutriments
            latent_dim=config.latent_dim,
        )

    def forward(self, name, steps, tags, categories, ingredient_ids, nutriments):
        name_encoded = self.text_encoder(name[0])

        step_embeddings = self.text_encoder(steps[0])
        steps_encoded = torch.mean(step_embeddings, dim=0)

        tags_encoded, categories_encoded = self.categorical_encoder(
            tags[0], categories[0]
        )

        ingredients_encoded = self.ingredient_encoder(ingredient_ids[0])

        nutriments_encoded = self.nutriment_encoder(nutriments[0])

        latent_embedding = self.fusion_layer(
            name_encoded.squeeze(),
            steps_encoded,
            tags_encoded,
            categories_encoded,
            ingredients_encoded,
            nutriments_encoded,
        )

        return latent_embedding
