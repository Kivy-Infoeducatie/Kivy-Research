from dataclasses import dataclass


@dataclass
class RecipeAutoencoderConfig:
    ingredient_vocab_size: int
    category_vocab_size: int
    tag_vocab_size: int
    num_nutriments: int
    ingredient_embedding_dim: int = 64
    categorical_embedding_dim: int = 32
    nutriment_embedding_dim: int = 64
    text_latent_dim: int = 128
    latent_dim: int = 256
    text_embedding_dim: int = 384
    text_transformer_model: str = "microsoft/MiniLM-L12-H384-uncased"
