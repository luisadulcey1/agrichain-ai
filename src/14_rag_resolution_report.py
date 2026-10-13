import json
import faiss
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - RAG Resolution Report
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load models and documents
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

ann_model = tf.keras.models.load_model(
    MODELS_DIR / "complaint_ann.keras"
)

encoder = joblib.load(
    MODELS_DIR / "category_encoder.joblib"
)

index = faiss.read_index(
    str(MODELS_DIR / "knowledge_base.index")
)

with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

# Example complaint
complaint = (
    "My agricultural supplies were supposed to arrive "
    "five days ago, but the shipment has not arrived. "
    "I need an updated delivery date and a solution."
)

# Generate embedding
embedding = embedding_model.encode(
    [complaint],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

# Classify complaint
probabilities = ann_model.predict(
    embedding,
    verbose=0
)[0]

label = int(np.argmax(probabilities))
category = encoder.inverse_transform([label])[0]
confidence = float(probabilities[label])

# Retrieve category-matched documents
scores, indices = index.search(
    embedding,
    index.ntotal
)

retrieved_docs = []

for score, idx in zip(scores[0], indices[0]):
    if idx < 0:
        continue

    doc = documents[int(idx)]

    if doc.get("category") == category:
        retrieved_docs.append({
            "doc_id": doc["doc_id"],
            "category": doc["category"],
            "doc_type": doc["doc_type"],
            "content": doc["content"],
            "similarity": round(float(score), 4)
        })

    if len(retrieved_docs) == 3:
        break

# Create structured report
report = {
    "complaint": complaint,
    "predicted_category": category,
    "classification_confidence": round(confidence, 4),
    "retrieved_documents": retrieved_docs,
    "status": "Pending investigation",
    "next_step": (
        "Review retrieved policies and verify the "
        "shipment details before proposing a resolution."
    )
}

# Save report
REPORTS_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = REPORTS_DIR / "rag_resolution_report.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=2, ensure_ascii=False)

# Display report
print("\nAGRiCHAIN - RAG RESOLUTION REPORT")

print("\nCOMPLAINT:")
print(report["complaint"])

print("\nPREDICTED CATEGORY:")
print(report["predicted_category"])

print("\nCLASSIFICATION CONFIDENCE:")
print(f"{confidence * 100:.2f}%")

print("\nRETRIEVED POLICIES:")

for doc in retrieved_docs:
    print(f"\n[{doc['doc_id']}]")
    print(doc["content"])

print("\nSTATUS:")
print(report["status"])

print("\nNEXT STEP:")
print(report["next_step"])

print("\nReport saved successfully!")