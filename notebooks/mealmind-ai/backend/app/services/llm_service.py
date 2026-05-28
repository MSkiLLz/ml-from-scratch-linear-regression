import ollama

from app.models.recipe_model import RecipeResponse
from app.services.json_utils import extract_json


MODEL_NAME = "llama3"


def generate_recipe(prompt: str):

    """
    Generate and validate recipe output.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional chef AI. "
                    "Always return ONLY valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_content = response["message"]["content"]

    print("\nRAW MODEL OUTPUT:\n")
    print(raw_content)

    # Extract JSON safely
    parsed_json = extract_json(raw_content)

    # Validate schema
    validated = RecipeResponse(**parsed_json)

    return validated.model_dump()