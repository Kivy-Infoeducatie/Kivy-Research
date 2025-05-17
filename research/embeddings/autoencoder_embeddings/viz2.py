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

class IngredientNameEncoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
    def forward(self, x):
        batch, n_ing, L = x.shape
        x = x.view(batch * n_ing, L)
        out = self.linear(x)
        out = F.relu(out)
        out = out.view(batch, n_ing, -1)
        return out

class DummyIngredientAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        batch, num_ingredients, in_dim = x.shape
        x = x.view(batch * num_ingredients, in_dim)
        out = self.linear(x)
        out = F.relu(out)
        out = out.view(batch, num_ingredients, -1)
        return out

class RecipeNutrimentEncoder(nn.Module):
    """
    Encodes recipe-level nutriments.
    """
    def __init__(self, recipe_nutriments_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(recipe_nutriments_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (batch, recipe_nutriments_dim)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class IngredientEncoder(nn.Module):
    """
    Encodes both ingredient-level and recipe-level nutriments.
    """
    def __init__(
        self,
        ingredient_nutriments_dim,
        recipe_nutriments_dim,
        quantity_dim,
        per_ing_hidden,
        per_ing_out,
        name_token_dim=6,
        name_emb_dim=64,
        fusion_hidden=128,
        fusion_out=128,
        aggregator_out=128,
        recipe_out_dim=64, # Output dim for recipe nutriment encoder
        final_out_dim=128,  # Output dim after full fusion
    ):
        super().__init__()
        self.nutriment_ffn = PerIngredientFFN(
            input_dim=ingredient_nutriments_dim + quantity_dim,
            hidden_dim=per_ing_hidden,
            out_dim=per_ing_out,
        )
        self.name_encoder = IngredientNameEncoder(
            input_dim=name_token_dim,
            out_dim=name_emb_dim,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(per_ing_out + name_emb_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, fusion_out),
            nn.ReLU(),
        )
        self.aggregator = DummyIngredientAggregator(
            input_dim=fusion_out,
            output_dim=aggregator_out,
        )
        self.recipe_encoder = RecipeNutrimentEncoder(
            recipe_nutriments_dim=recipe_nutriments_dim,
            out_dim=recipe_out_dim,
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(aggregator_out + recipe_out_dim, final_out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        ingredient_nutriments,         # [batch, num_ingredients, ingredient_nutriments_dim]
        ingredient_quantities,         # [batch, num_ingredients, quantity_dim]
        ingredient_name_input,         # [batch, num_ingredients, name_token_dim]
        recipe_nutriments,             # [batch, recipe_nutriments_dim]
        ingredient_name_attention_mask=None, # unused
    ):
        batch, num_ingredients, _ = ingredient_nutriments.shape
        x_structured = torch.cat([ingredient_nutriments, ingredient_quantities], dim=-1)
        x_structured = self.nutriment_ffn(x_structured)
        name_emb = self.name_encoder(ingredient_name_input)
        fused = torch.cat([x_structured, name_emb], dim=-1)
        fused = fused.view(batch * num_ingredients, -1)
        fused = self.fusion_mlp(fused)
        fused = fused.view(batch, num_ingredients, -1)
        encoded_seq = self.aggregator(fused)
        pooled_ingredients = encoded_seq.mean(dim=1)  # (batch, aggregator_out)
        encoded_recipe = self.recipe_encoder(recipe_nutriments)  # (batch, recipe_out_dim)
        full_fused = torch.cat([pooled_ingredients, encoded_recipe], dim=-1)
        final_output = self.final_fusion(full_fused)  # (batch, final_out_dim)
        return final_output

# Example usage:
encoder = IngredientEncoder(
    ingredient_nutriments_dim=9,
    recipe_nutriments_dim=6,
    quantity_dim=1,
    per_ing_hidden=32,
    per_ing_out=32,
    name_token_dim=6,
    name_emb_dim=64,
    fusion_hidden=128,
    fusion_out=128,
    aggregator_out=128,
    recipe_out_dim=64,
    final_out_dim=128,
)

batch = 2
num_ingredients = 5
ingredient_nutriments = torch.randn(batch, num_ingredients, 9)
ingredient_quantities = torch.randn(batch, num_ingredients, 1)
ingredient_name_input = torch.randn(batch, num_ingredients, 6)
recipe_nutriments = torch.randn(batch, 6)

pooled_vector = encoder(ingredient_nutriments, ingredient_quantities, ingredient_name_input, recipe_nutriments)
print(pooled_vector.shape)  # (batch, final_out_dim)

torch.onnx.export(
    encoder,
    (ingredient_nutriments, ingredient_quantities, ingredient_name_input, recipe_nutriments, None),
    "ingredient_block_with_recipe_nutriments.onnx",
    input_names=[
        "ingredient_nutriments", "ingredient_quantities", "ingredient_name_input",
        "recipe_nutriments", "ingredient_name_attention_mask"
    ],
    output_names=["output"],
    dynamic_axes={
        "ingredient_nutriments": {0: "batch_size"},
        "ingredient_quantities": {0: "batch_size"},
        "ingredient_name_input": {0: "batch_size"},
        "recipe_nutriments": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)