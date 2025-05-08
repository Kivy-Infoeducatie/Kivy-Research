import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        transformer_model="microsoft/MiniLM-L12-H384-uncased",
        text_embedding_dim=128,
    ):
        super(TextEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.fc_text = nn.Linear(
            384, text_embedding_dim
        )  # MiniLM produces 384-dimensional embeddings

    def forward(self, text_input):
        # Encode text using MiniLM
        text_output = self.transformer(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
        )
        text_embedding = self.fc_text(
            text_output.pooler_output
        )  # Shape: (batch_size, text_embedding_dim)
        return text_embedding
