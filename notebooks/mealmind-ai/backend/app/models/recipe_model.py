from pydantic import BaseModel
from typing import List


class RecipeResponse(BaseModel):

    recipe_name: str
    region: str
    estimated_cost: int
    prep_time: str

    ingredients: List[str]

    instructions: List[str]

    tips: List[str]