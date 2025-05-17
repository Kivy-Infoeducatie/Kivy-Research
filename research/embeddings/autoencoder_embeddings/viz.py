import torch
import torch.nn as nn
import torch.nn.functional as F

class PerIngredientFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch, n_ing, _ = x.shape
        x = x.view(batch * n_ing, -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.view(batch, n_ing, -1)
        return x

if __name__ == "__main__":
    input_dim = 9
    hidden_dim = 32
    output_dim = 16

    model = PerIngredientFFN(input_dim, hidden_dim, output_dim)
    model.eval()  # Set the model to evaluation mode.

    # Use a 3D dummy input: (batch_size, n_ing, input_dim)
    dummy_input = torch.randn(1, 4, input_dim)  # For example, 1 batch, 4 ingredients, 9 features each

    torch.onnx.export(
        model,
        dummy_input,
        "nutriment_block.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "n_ing"},
            "output": {0: "batch_size", 1: "n_ing"},
        },
    )

