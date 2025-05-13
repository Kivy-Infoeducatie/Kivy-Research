import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import os

from embeddings.autoencoder_embeddings.recipe_autoencoder import RecipeAutoencoder
from embeddings.autoencoder_embeddings.recipe_dataset import RecipeDataset

# Device setup
device = torch.device("cpu")

# Model and training setup
model = RecipeAutoencoder(
    num_nutriments=9,
    tag_vocab_size=281,
    category_vocab_size=262,
    ingredient_vocab_size=20860,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5


# Dataset
def passthrough_collate_fn(batch):
    # Returns the batch as-is: list of samples (tuples)
    return tuple(zip(*batch))


data = pd.read_pickle("embeddings_train_processed_data.pkl")
train_dataset = RecipeDataset(data)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=passthrough_collate_fn
)

# 🔽 Create a directory to store checkpoints
os.makedirs("checkpoints", exist_ok=True)

# Print whether gradients are enabled for each parameter
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    running_loss = 0.0

    progress_bar = tqdm(
        enumerate(train_loader),
        desc=f"Epoch {epoch+1}/{num_epochs}",
        total=len(train_loader),
    )

    for batch_idx, batch in progress_bar:
        batch = [x for x in batch]
        name, steps, tags, categories, ingredient_ids, nutriments = batch

        optimizer.zero_grad()

        (
            latent_embedding,
            text_decoded,
            tags_decoded,
            categories_decoded,
            ingredients_decoded,
            nutriments_decoded,
        ) = model(name, steps, tags, categories, ingredient_ids, nutriments)

        loss = criterion(nutriments_decoded, nutriments[0])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        average_loss = running_loss / (batch_idx + 1)

        # Update the progress bar with the latest loss and running average loss
        progress_bar.set_postfix(
            current_loss=f"{loss.item():.4f}", avg_loss=f"{average_loss:.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    torch.save(checkpoint, f"checkpoints/epoch_{epoch+1:02d}.pth")
    print(f"Checkpoint saved: checkpoints/epoch_{epoch+1:02d}.pth")
