import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
INPUT_PATH = Path("../data/processed/chunks.json")
MODEL_DIR = Path("../model")
INDEX_PATH = MODEL_DIR / "recipe_index.faiss"
METADATA_PATH = MODEL_DIR / "chunk_metadata.json"

# Ensure output directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load chunks
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract text to embed
texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks.")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# Determine embedding dimension
dimension = embeddings.shape[1]
print(f"Embedding dimension: {dimension}")

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

# Save index
faiss.write_index(index, str(INDEX_PATH))

# Save chunk metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Saved FAISS index to {INDEX_PATH}")
print(f"Saved metadata to {METADATA_PATH}")
print("Embedding creation complete.")