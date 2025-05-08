import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from embeddings.autoencoder_embeddings.recipe_autoencoder import RecipeAutoencoder


class RecipeDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        tag_vocab,
        category_vocab,
        ingredient_vocab,
        max_length=128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_vocab = tag_vocab
        self.category_vocab = category_vocab
        self.ingredient_vocab = ingredient_vocab
        self.max_length = max_length

    def encode_strings(self, strings, vocab):
        return [vocab.get(s, 0) for s in strings]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data[idx]

        name = self.tokenizer(
            recipe["name"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        description = self.tokenizer(
            recipe["description"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        steps = [
            self.tokenizer(
                step,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for step in recipe["steps"]
        ]
        tags = torch.tensor(
            self.encode_strings(recipe["tags"], self.tag_vocab), dtype=torch.long
        )
        ingredients = torch.tensor(
            self.encode_strings(recipe["ingredients"], self.ingredient_vocab),
            dtype=torch.long,
        )
        category = torch.tensor(
            self.category_vocab.get(recipe["category"], 0), dtype=torch.long
        )
        nutriments = torch.tensor(recipe["nutriments"], dtype=torch.float32)

        return name, description, steps, tags, category, ingredients, nutriments


def build_vocab(items, min_freq=1):
    if isinstance(items[0], list):
        items = [item for sublist in items for item in sublist]
    counter = Counter(items)
    vocab = {
        word: idx + 1
        for idx, (word, count) in enumerate(counter.items())
        if count >= min_freq
    }
    vocab["<UNK>"] = 0
    return vocab


def train_minilm_autoencoder(
    train_data,
    val_data,
    epochs=10,
    batch_size=64,
    max_length=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print("🔧 Building vocabularies...")
    tag_vocab = build_vocab([recipe["tags"] for recipe in train_data], min_freq=1)
    category_vocab = build_vocab(
        [recipe["category"] for recipe in train_data], min_freq=1
    )
    ingredient_vocab = build_vocab(
        [recipe["ingredients"] for recipe in train_data], min_freq=1
    )

    print("📖 Initializing tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    train_dataset = RecipeDataset(
        train_data,
        tokenizer,
        tag_vocab,
        category_vocab,
        ingredient_vocab,
        max_length=max_length,
    )
    val_dataset = RecipeDataset(
        val_data,
        tokenizer,
        tag_vocab,
        category_vocab,
        ingredient_vocab,
        max_length=max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("🧠 Initializing model...")
    model = RecipeAutoencoder(
        latent_dim=256, ingredient_vocab_size=len(ingredient_vocab)
    ).to(device)
    criterion_text = nn.MSELoss()
    criterion_tags = nn.BCEWithLogitsLoss()
    criterion_category = nn.CrossEntropyLoss()
    criterion_ingredients = nn.BCEWithLogitsLoss()
    criterion_nutriments = nn.MSELoss()
    transformer_params = list(model.text_encoder.transformer.parameters())
    other_params = [p for p in model.parameters() if p not in transformer_params]

    optimizer = torch.optim.Adam([
        {"params": transformer_params, "lr": 1e-5},
        {"params": other_params, "lr": 1e-4}
    ])

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            name, description, steps, tags, category, ingredients, nutriments = batch
            name = {key: val.to(device) for key, val in name.items()}
            description = {key: val.to(device) for key, val in description.items()}
            steps = [
                {key: val.to(device) for key, val in step.items()} for step in steps
            ]
            tags, category, ingredients, nutriments = (
                tags.to(device),
                category.to(device),
                ingredients.to(device),
                nutriments.to(device),
            )

            (
                latent_embedding,
                text_decoded,
                tags_decoded,
                category_decoded,
                ingredients_decoded,
                nutriments_decoded,
            ) = model(name, description, steps, tags, category, ingredients, nutriments)

            loss_text = criterion_text(text_decoded, name["input_ids"].float())
            loss_tags = criterion_tags(
                tags_decoded,
                nn.functional.one_hot(tags, num_classes=len(tag_vocab)).float(),
            )
            loss_category = criterion_category(category_decoded, category)
            loss_ingredients = criterion_ingredients(
                ingredients_decoded,
                nn.functional.one_hot(
                    ingredients, num_classes=len(ingredient_vocab)
                ).float(),
            )
            loss_nutriments = criterion_nutriments(nutriments_decoded, nutriments)

            loss = (
                loss_text
                + loss_tags
                + loss_category
                + loss_ingredients
                + loss_nutriments
            )
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                name, description, steps, tags, category, ingredients, nutriments = (
                    batch
                )
                name = {key: val.to(device) for key, val in name.items()}
                description = {key: val.to(device) for key, val in description.items()}
                steps = [
                    {key: val.to(device) for key, val in step.items()} for step in steps
                ]
                tags, category, ingredients, nutriments = (
                    tags.to(device),
                    category.to(device),
                    ingredients.to(device),
                    nutriments.to(device),
                )

                (
                    latent_embedding,
                    text_decoded,
                    tags_decoded,
                    category_decoded,
                    ingredients_decoded,
                    nutriments_decoded,
                ) = model(
                    name, description, steps, tags, category, ingredients, nutriments
                )

                loss_text = criterion_text(text_decoded, name["input_ids"].float())
                loss_tags = criterion_tags(
                    tags_decoded,
                    nn.functional.one_hot(tags, num_classes=len(tag_vocab)).float(),
                )
                loss_category = criterion_category(category_decoded, category)
                loss_ingredients = criterion_ingredients(
                    ingredients_decoded,
                    nn.functional.one_hot(
                        ingredients, num_classes=len(ingredient_vocab)
                    ).float(),
                )
                loss_nutriments = criterion_nutriments(nutriments_decoded, nutriments)

                loss = (
                    loss_text
                    + loss_tags
                    + loss_category
                    + loss_ingredients
                    + loss_nutriments
                )
                total_val_loss += loss.item()

        print(
            f"📊 Epoch [{epoch + 1}/{epochs}] - Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}"
        )

    print("💾 Saving model...")
    torch.save(model.state_dict(), "minilm_recipe_autoencoder.pth")
    print("✅ Model training complete and saved as 'minilm_recipe_autoencoder.pth'.")


# Load data and train the model
print("📂 Loading dataset...")
df = pd.read_pickle("embeddings_train_data.pkl")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_data = train_df.to_dict(orient="records")
val_data = val_df.to_dict(orient="records")

train_minilm_autoencoder(
    train_data=train_data,
    val_data=val_data,
    epochs=10,
    batch_size=64,
    max_length=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
