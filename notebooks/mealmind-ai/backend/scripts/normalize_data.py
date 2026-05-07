import json
from pathlib import Path

RAW_PATH = Path("../data/raw/full_dataset.json")
OUTPUT_PATH = Path("../data/processed/normalized_recipes.json")

INGREDIENT_MAPPINGS = {
    "garlic cloves": "garlic",
    "fresh garlic": "garlic",
    "minced garlic": "garlic",
    "white rice": "rice",
    "brown rice": "rice"
}


def normalize_ingredient(ingredient):
    ingredient = ingredient.lower().strip()

    if ingredient in INGREDIENT_MAPPINGS:
        return INGREDIENT_MAPPINGS[ingredient]

    return ingredient


with open(RAW_PATH) as f:
    recipes = json.load(f)

normalized = []

for recipe in recipes:

    normalized_ingredients = [
        normalize_ingredient(i)
        for i in recipe["ingredients"]
    ]

    recipe["ingredients"] = normalized_ingredients

    normalized.append(recipe)

with open(OUTPUT_PATH, "w") as f:
    json.dump(normalized, f, indent=2)

print("Normalization complete.")