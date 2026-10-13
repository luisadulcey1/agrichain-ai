import json
import faiss
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sentence_transformers import SentenceTransformer

# AgriChain AI - Improved RAG Retrieval

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load FAISS index
index = faiss.read_index(
    str(MODELS_DIR / "knowledge_base.index")
)

# Load knowledge base metadata
with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

# Load MiniLM
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load ANN and category encoder
ann_model = tf.keras.models.load_model(
    MODELS_DIR / "complaint_ann.keras"
)

encoder = joblib.load(
    MODELS_DIR / "category_encoder.joblib"
)

# Customer complaint
complaint = (
    "My agricultural supplies were supposed to arrive "
    "five days ago, but the shipment has not arrived. "
    "I need an updated delivery date and a solution."
)

print("\nAGRiCHAIN - IMPROVED RAG RETRIEVAL")
print("\nCUSTOMER COMPLAINT:")
print(complaint)

# Generate complaint embedding
embedding = embedding_model.encode(
    [complaint],
    convert_to_numpy=True,
    normalize_embeddings=True
)

embedding = np.asarray(embedding, dtype="float32")

# Predict category using ANN
probabilities = ann_model.predict(
    embedding,
    verbose=0
)[0]

predicted_label = int(np.argmax(probabilities))
predicted_category = encoder.inverse_transform(
    [predicted_label]
)[0]

confidence = float(probabilities[predicted_label])

print("\nANN PREDICTION")
print("Category:", predicted_category)
print(f"Confidence: {confidence * 100:.2f}%")

# Search entire FAISS index, then filter by category
scores, indices = index.search(
    embedding,
    index.ntotal
)

filtered_results = []

for score, doc_index in zip(scores[0], indices[0]):
    if doc_index < 0:
        continue

    doc = documents[int(doc_index)]

    if doc.get("category") == predicted_category:
        filtered_results.append((float(score), doc))

    if len(filtered_results) == 3:
        break

print("\nTOP 3 CATEGORY-MATCHED DOCUMENTS")

for rank, (score, doc) in enumerate(
    filtered_results,
    start=1
):
    print("\n" + "=" * 60)
    print(f"RESULT {rank}")
    print("=" * 60)

    print("Document ID:", doc.get("doc_id"))
    print("Category:", doc.get("category"))
    print("Document Type:", doc.get("doc_type"))
    print("Similarity Score:", round(score, 4))

    print("\nCONTENT:")
    print(doc.get("content"))

if not filtered_results:
    print(
        "\nNo category-matched documents found. "
        "A broader search may be needed."
    )

print("\nImproved retrieval completed successfully!")