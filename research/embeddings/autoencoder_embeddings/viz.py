import torch
import torch.nn as nn
import torch.nn.functional as F


class NutrimentBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(NutrimentBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


input_dim = 9
hidden_dim = 32
output_dim = 16

model = NutrimentBlock(input_dim, hidden_dim, output_dim)
model.eval()  # Set the model to evaluation mode.

dummy_input = torch.randn(1, input_dim)

torch.onnx.export(
    model,
    dummy_input,
    "nutriment_block.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
