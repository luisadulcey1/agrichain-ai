import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - Task 2
# Build FAISS Knowledge Base Index
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load knowledge base
with open(
    RAW_DIR / "knowledge_base.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

print("\nAGRiCHAIN - BUILDING FAISS INDEX")
print("\nOriginal documents:", len(documents))

# Prepare text for semantic search
texts = []

for doc in documents:
    text = (
        f"Category: {doc.get('category', '')}. "
        f"Document type: {doc.get('doc_type', '')}. "
        f"{doc.get('content', '')}"
    )

    texts.append(text)

# Load MiniLM
print("\nLoading MiniLM...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Generate document embeddings
print("\nGenerating knowledge base embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

# Create FAISS index using cosine similarity
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save FAISS index
faiss.write_index(
    index,
    str(MODELS_DIR / "knowledge_base.index")
)

# Save metadata in the same order as vectors
with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        documents,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nFAISS INDEX RESULTS")
print("Documents indexed:", index.ntotal)
print("Embedding dimension:", dimension)

print("\nFAISS index saved successfully!")