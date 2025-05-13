import torch
from torch.utils.data import Dataset

# Set MPS device if available, fallback to CPU
device = torch.device("cpu")


class RecipeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data.iloc[idx]

        # Move tensor-like fields to MPS
        tags = torch.tensor(recipe["tags"], dtype=torch.int64).to(device)
        category = torch.tensor(recipe["category"], dtype=torch.int64).to(device)
        ingredients = torch.tensor(recipe["ingredients"], dtype=torch.int64).to(device)
        nutriments = torch.tensor(recipe["nutriments"], dtype=torch.float32).to(device)

        return (
            recipe["name"],  # str
            recipe["steps"],  # str or tokenized text
            tags,  # tensor on MPS
            category,  # tensor on MPS
            ingredients,  # tensor on MPS
            nutriments,  # tensor on MPS
        )
