from torch import nn

from embeddings.autoencoder_embeddings.recipe_autoencoder_config import RecipeAutoencoderConfig


class RecipeDecoder(nn.Module):
    def __init__(self, config: RecipeAutoencoderConfig):
        super(RecipeDecoder, self).__init__()

        self.latent_to_text = nn.Linear(config.latent_dim, config.text_embedding_dim)

        self.latent_to_tags = nn.Linear(config.latent_dim, config.tag_vocab_size)

        self.latent_to_categories = nn.Linear(
            config.latent_dim, config.category_vocab_size
        )

        self.latent_to_ingredients = nn.Linear(
            config.latent_dim, config.ingredient_vocab_size
        )

        self.latent_to_nutriments = nn.Linear(config.latent_dim, config.num_nutriments)

    def forward(self, latent_embedding):
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
