import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - Task 2
# Test FAISS Semantic Retrieval
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load FAISS index
index = faiss.read_index(
    str(MODELS_DIR / "knowledge_base.index")
)

# Load document metadata
with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

# Load the same embedding model used to build the index
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Example customer complaint
complaint = (
    "My agricultural supplies were supposed to arrive "
    "five days ago, but the shipment has not arrived. "
    "I need an updated delivery date and a solution."
)

print("\nAGRiCHAIN - FAISS RETRIEVAL TEST")
print("\nCUSTOMER COMPLAINT:")
print(complaint)

# Convert complaint to embedding
query_embedding = model.encode(
    [complaint],
    convert_to_numpy=True,
    normalize_embeddings=True
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)

# Retrieve top 3 documents
scores, indices = index.search(query_embedding, 3)

print("\nTOP 3 RETRIEVED DOCUMENTS")

for rank, (score, doc_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    if doc_index < 0:
        continue

    doc = documents[int(doc_index)]

    print("\n" + "=" * 60)
    print(f"RESULT {rank}")
    print("=" * 60)

    print("Document ID:", doc.get("doc_id"))
    print("Category:", doc.get("category"))
    print("Document Type:", doc.get("doc_type"))
    print("Similarity Score:", round(float(score), 4))
    print("\nCONTENT:")
    print(doc.get("content"))

print("\nRetrieval test completed successfully!")