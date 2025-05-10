import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        transformer_model: str,
        transformer_model_embedding_dim: int,
        text_embedding_dim: int,
    ):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.fc_text = nn.Linear(transformer_model_embedding_dim, text_embedding_dim)

    def forward(self, text_batch):
        print(text_batch)
        print(123)
        encoded = self.tokenizer(
            text_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(self.transformer.device)

        output = self.transformer(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )

        cls_embedding = output.last_hidden_state[:, 0, :]  # [CLS] token
        text_embedding = self.fc_text(cls_embedding)
        return text_embedding
