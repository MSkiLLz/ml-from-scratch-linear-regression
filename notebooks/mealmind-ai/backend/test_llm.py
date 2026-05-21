from app.rag.prompt_builder import build_prompt
from app.services.llm_service import generate_recipe

retrieved_chunks = [
    {
        "text": """
Recipe: Jollof Rice
Region: West African
Ingredients: rice, tomatoes, onions, pepper
Estimated Cost: $12
Instructions: Cook tomato sauce. Add rice and simmer.
"""
    }
]

prompt = build_prompt(
    ingredients=["rice", "chicken", "garlic"],
    budget=15,
    region="West African",
    retrieved_chunks=retrieved_chunks
)

recipe = generate_recipe(prompt)

print(recipe)