from fastapi import FastAPI

from app.models.request_model import RecipeRequest

from app.rag.retriever import RecipeRetriever
from app.rag.prompt_builder import build_prompt

from app.services.llm_service import generate_recipe


app = FastAPI(
    title="MealMind AI",
    description="AI-powered RAG meal generation system",
    version="1.0.0"
)

# Load retriever once at startup
retriever = RecipeRetriever()


@app.get("/")
def home():

    return {
        "message": "MealMind AI Backend Running"
    }


@app.post("/generate-recipe")
def generate_recipe_endpoint(request: RecipeRequest):

    # Step 1 — Retrieve relevant recipes
    retrieved_chunks = retriever.retrieve(
        ingredients=request.ingredients,
        budget=request.budget,
        region=request.region,
        k=3
    )

    # Step 2 — Build prompt
    prompt = build_prompt(
        ingredients=request.ingredients,
        budget=request.budget,
        region=request.region,
        retrieved_chunks=retrieved_chunks
    )

    # Step 3 — Generate recipe
    recipe = generate_recipe(prompt)

    return {
        "success": True,
        "recipe": recipe
    }