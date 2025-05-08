import torch
import torch.nn as nn
from text_encoder import TextEncoder
from categorical_encoder import CategoricalEncoder
from ingredient_encoder import IngredientEncoder
from nutriment_encoder import NutrimentEncoder
from fusion_layer import FusionLayer


class RecipeAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, ingredient_vocab_size=10000):
        super(RecipeAutoencoder, self).__init__()

        # Submodules
        self.text_encoder = TextEncoder(
            transformer_model="microsoft/MiniLM-L12-H384-uncased",
            text_embedding_dim=128,
        )
        self.categorical_encoder = CategoricalEncoder(
            num_tags=305, num_categories=300, embedding_dim=32
        )
        self.ingredient_encoder = IngredientEncoder(
            ingredient_vocab_size=ingredient_vocab_size, embedding_dim=64
        )
        self.nutriment_encoder = NutrimentEncoder()
        self.fusion_layer = FusionLayer(
            input_dim=128 + 128 + 128 + 32 + 32 + 64 + 64, latent_dim=latent_dim
        )

        # Decoders
        self.latent_to_text = nn.Linear(
            latent_dim, 384
        )  # For MiniLM embedding reconstruction
        self.latent_to_tags = nn.Linear(latent_dim, 305)
        self.latent_to_categories = nn.Linear(latent_dim, 300)
        self.latent_to_ingredients = nn.Linear(latent_dim, ingredient_vocab_size)
        self.latent_to_nutriments = nn.Linear(latent_dim, 9)

    def forward(
        self, name, description, steps, tags, categories, ingredient_ids, nutriments
    ):
        # Encode Text Features
        name_encoded = self.text_encoder(name)
        description_encoded = self.text_encoder(description)

        # Aggregate Steps (list of strings)
        step_embeddings = [self.text_encoder(step) for step in steps]
        steps_encoded = torch.mean(
            torch.stack(step_embeddings, dim=0), dim=0
        )  # Mean pooling on steps

        # Encode Tags, Categories
        tags_encoded, categories_encoded = self.categorical_encoder(tags, categories)

        # Encode Ingredients
        ingredients_encoded = self.ingredient_encoder(ingredient_ids)

        # Encode Nutriments
        nutriments_encoded = self.nutriment_encoder(nutriments)

        # Fuse All Features
        latent_embedding = self.fusion_layer(
            name_encoded,
            description_encoded,
            steps_encoded,
            tags_encoded,
            categories_encoded,
            ingredients_encoded,
            nutriments_encoded,
        )

        # Decode Features
        text_decoded = self.latent_to_text(latent_embedding)
        tags_decoded = self.latent_to_tags(latent_embedding)
        categories_decoded = self.latent_to_categories(latent_embedding)
        ingredients_decoded = self.latent_to_ingredients(latent_embedding)
        nutriments_decoded = self.latent_to_nutriments(latent_embedding)

        return (
            latent_embedding,
            text_decoded,
            tags_decoded,
            categories_decoded,
            ingredients_decoded,
            nutriments_decoded,
        )
