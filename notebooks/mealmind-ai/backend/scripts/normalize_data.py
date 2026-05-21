import json
from pathlib import Path

RAW_PATH = Path("../data/raw/haitian_recipes.json")
OUTPUT_PATH = Path("../data/processed/normalized_haitian_recipes.json")

# Ingredient normalization mappings for Haitian recipes
# Keys should be lowercase phrases that may appear in ingredient strings.
# Values are the canonical ingredient names used by the RAG system.

INGREDIENT_MAPPINGS = {
    # Garlic
    "garlic cloves": "garlic",
    "cloves garlic": "garlic",
    "fresh garlic": "garlic",
    "minced garlic": "garlic",
    "garlic, minced": "garlic",

    # Rice
    "white rice": "rice",
    "brown rice": "rice",
    "long grain rice": "rice",

    # Onions
    "red onion": "onion",
    "white onion": "onion",
    "yellow onion": "onion",
    "sliced onion": "onion",
    "chopped onion": "onion",

    # Peppers
    "green bell pepper": "bell pepper",
    "red bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "scotch bonnet peppers": "scotch bonnet pepper",
    "hot peppers": "scotch bonnet pepper",

    # Citrus
    "limes": "lime",
    "lime juice": "lime",
    "lemons": "lemon",
    "lemon juice": "lemon",
    "orange juice": "orange",

    # Herbs and spices
    "fresh thyme": "thyme",
    "dried thyme": "thyme",
    "black pepper": "pepper",
    "ground black pepper": "pepper",

    # Oils and fats
    "vegetable oil": "oil",
    "olive oil": "oil",
    "cooking oil": "oil",
    "oil for frying": "oil",

    # Proteins
    "pork shoulder": "pork",
    "pork shoulder, cut into cubes": "pork",
    "beef stew meat": "beef",
    "goat meat": "goat",
    "crab meat": "crab",

    # Beans and legumes
    "black beans": "beans",
    "red beans": "beans",
    "kidney beans": "beans",

    # Vegetables
    "calabaza squash": "pumpkin",
    "pumpkin or calabaza squash": "pumpkin",
    "sweet potatoes": "sweet potato",
    "green plantains": "plantain",
    "plantains": "plantain",
    "cabbage wedge": "cabbage",

    # Grains and starches
    "cornmeal": "cornmeal",
    "pasta": "pasta",

    # Specialty Haitian ingredients
    "djon djon mushrooms": "djon djon mushroom",
    "dried djon djon mushrooms": "djon djon mushroom",
    "pikliz": "pikliz",

    # Liquids
    "coconut milk": "coconut milk",
    "white vinegar": "vinegar",
    "vinegar": "vinegar",

    # Sweeteners
    "brown sugar": "sugar",
    "white sugar": "sugar",

    # Miscellaneous
    "raisins": "raisin",
    "vanilla extract": "vanilla",
    "tomato paste": "tomato paste",
    "green onions": "green onion"
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

    normalized_ingredients = list(dict.fromkeys(
        normalize_ingredient(i)
        for i in recipe["ingredients"]
    ))

    recipe["ingredients"] = normalized_ingredients

    normalized.append(recipe)

with open(OUTPUT_PATH, "w") as f:
    json.dump(normalized, f, indent=2)

print("Normalization complete.")