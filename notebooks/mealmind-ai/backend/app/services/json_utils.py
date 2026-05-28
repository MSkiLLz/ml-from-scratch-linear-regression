import json
import re


def extract_json(text: str):

    """
    Extract JSON object from LLM response text.
    """

    # Remove markdown fences
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found.")

    json_text = match.group(0)

    return json.loads(json_text)