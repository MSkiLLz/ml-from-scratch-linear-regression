import json
from pathlib import Path

INPUT_PATH = Path("../data/processed/enriched_recipes.json")
OUTPUT_PATH = Path("../data/processed/chunks.json")

with open(INPUT_PATH) as f:
    recipes = json.load(f)

chunks = []

for recipe in recipes:

    chunk = f"""
    Recipe: {recipe['title']}

    Region: {recipe['region']}

    Ingredients:
    {", ".join(recipe['ingredients'])}

    Estimated Cost:
    ${recipe['estimated_cost']}

    Instructions:
    {recipe['instructions']}
    """

    chunks.append({
        "text": chunk,
        "metadata": {
            "title": recipe["title"],
            "region": recipe["region"],
            "estimated_cost": recipe["estimated_cost"]
        }
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(chunks, f, indent=2)

print("Chunks created.")