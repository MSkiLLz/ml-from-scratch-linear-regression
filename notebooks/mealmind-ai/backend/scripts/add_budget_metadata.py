import json
from pathlib import Path

INPUT_PATH = Path("../data/processed/normalized_recipes.json")
OUTPUT_PATH = Path("../data/processed/enriched_recipes.json")

INGREDIENT_PRICES = {
    "rice": 2,
    "chicken": 5,
    "garlic": 1,
    "egg": 2,
    "tomatoes": 3,
    "pepper": 2,
    "soy sauce": 3,
    "onions": 2,
    "chicken stock": 3
}


def estimate_cost(ingredients):

    total = 0

    for ingredient in ingredients:
        total += INGREDIENT_PRICES.get(ingredient, 2)

    return total


with open(INPUT_PATH) as f:
    recipes = json.load(f)

for recipe in recipes:

    recipe["estimated_cost"] = estimate_cost(
        recipe["ingredients"]
    )

with open(OUTPUT_PATH, "w") as f:
    json.dump(recipes, f, indent=2)

print("Budget metadata added.")