import json


def build_prompt(ingredients, budget, region, retrieved_chunks):
    """
    Build a prompt for the LLM using user constraints and retrieved recipe knowledge.
    """

    # Normalize ingredient input
    if isinstance(ingredients, list):
        ingredients_text = ", ".join(ingredients)
    else:
        ingredients_text = str(ingredients)

    # Combine retrieved chunk text
    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
You are an expert chef AI specializing in realistic, budget-conscious recipes.

USER REQUIREMENTS
-----------------
Ingredients Available: {ingredients_text}
Maximum Budget: ${budget}
Cuisine Region: {region}

RETRIEVED RECIPE KNOWLEDGE
--------------------------
{context}

INSTRUCTIONS
------------
Create one original recipe inspired by the retrieved knowledge.

Rules:
1. Respect the user's ingredient list.
2. Stay within the specified budget.
3. Match the requested cuisine region.
4. You may use basic pantry staples (salt, pepper, oil, water).
5. Provide output as valid JSON only.

Required JSON format:
{{
  "recipe_name": "...",
  "region": "{region}",
  "estimated_cost": 0,
  "prep_time": "...",
  "ingredients": ["..."],
  "instructions": ["Step 1", "Step 2"],
  "tips": ["..."]
}}
"""
    return prompt