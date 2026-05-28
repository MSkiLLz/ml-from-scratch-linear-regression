from pydantic import BaseModel
from typing import List


class RecipeRequest(BaseModel):

    ingredients: List[str]

    budget: int

    region: str