import os
import json
import ollama

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
# load_dotenv()

# Initialize client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_recipe(prompt: str) -> dict:
    """
    Sends the prompt to the LLM and returns parsed JSON.
    """

    response =  ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional chef AI. "
                    "Always return valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response["message"]["content"]
    
    # Remove markdown code fences if present
    content = content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]

        if content.startswith("json"):
            content = content[4:]

    content = content.strip()

    # Convert JSON string into Python dictionary
    return json.loads(content)