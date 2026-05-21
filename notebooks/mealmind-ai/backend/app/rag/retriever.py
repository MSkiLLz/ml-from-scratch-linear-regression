import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Paths relative to this file:
# backend/app/rag/retriever.py
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "model"

INDEX_PATH = MODEL_DIR / "recipe_index.faiss"
METADATA_PATH = MODEL_DIR / "chunk_metadata.json"


class RecipeRetriever:
    def __init__(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading metadata...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Retriever ready.")

    def retrieve(self, ingredients, budget=None, region=None, k=5):
        """
        ingredients: list[str] or comma-separated string
        budget: maximum allowed estimated cost
        region: desired cuisine region
        k: number of candidates to return after filtering
        """

        # Normalize ingredient input
        if isinstance(ingredients, list):
            query = ", ".join(ingredients)
        else:
            query = str(ingredients)

        # Embed the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # Search more than k to allow filtering
        search_k = min(len(self.chunks), max(k * 3, 10))
        distances, indices = self.index.search(query_embedding, search_k)

        results = []

        for idx in indices[0]:
            chunk = self.chunks[idx]
            metadata = chunk["metadata"]

            # Region filter
            if region:
                if metadata["region"].lower() != region.lower():
                    continue

            # Budget filter
            if budget is not None:
                if metadata["estimated_cost"] > budget:
                    continue

            results.append(chunk)

            if len(results) >= k:
                break

        return results


# Quick test when run directly
if __name__ == "__main__":
    retriever = RecipeRetriever()

    matches = retriever.retrieve(
        ingredients=["rice", "chicken"],
        budget=15,
        region="West African",
        k=3
    )

    for i, match in enumerate(matches, start=1):
        print(f"\n=== Match {i} ===")
        print(match["metadata"])
        print(match["text"])