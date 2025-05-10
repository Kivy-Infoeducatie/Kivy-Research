from torch.utils.data import Dataset


class RecipeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data.iloc[idx]

        print(recipe)

        return (
            recipe["name"],
            recipe["steps"],
            recipe["tags"],
            recipe["category"],
            recipe["ingredients"],
            recipe["nutriments"],
        )
