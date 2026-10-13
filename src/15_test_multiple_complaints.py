import json
import faiss
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - Multiple Complaint Retrieval Tests
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

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

# Test complaints and expected categories
test_cases = [
    {
        "complaint": (
            "Several boxes arrived broken and the products "
            "inside were damaged during transportation."
        ),
        "expected": "Damaged Goods"
    },
    {
        "complaint": (
            "My invoice shows a higher price than the "
            "amount agreed upon when I placed the order."
        ),
        "expected": "Pricing Error"
    },
    {
        "complaint": (
            "Some products are missing from my shipment. "
            "I received fewer items than I ordered."
        ),
        "expected": "Missing Items"
    },
    {
        "complaint": (
            "The shipment arrived five days late and "
            "I need an updated delivery explanation."
        ),
        "expected": "Delivery Delay"
    }
]

print("\nAGRiCHAIN - MULTIPLE COMPLAINT TESTS")

correct_classifications = 0
relevant_retrievals = 0

for number, case in enumerate(test_cases, start=1):

    complaint = case["complaint"]
    expected = case["expected"]

    embedding = embedding_model.encode(
        [complaint],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    probabilities = ann_model.predict(
        embedding,
        verbose=0
    )[0]

    label = int(np.argmax(probabilities))
    category = encoder.inverse_transform([label])[0]
    confidence = float(probabilities[label])

    classification_correct = category == expected

    if classification_correct:
        correct_classifications += 1

    scores, indices = index.search(
        embedding,
        index.ntotal
    )

    matched_documents = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        doc = documents[int(idx)]

        if doc.get("category") == category:
            matched_documents.append(doc)

        if len(matched_documents) == 3:
            break

    # Basic category-level retrieval check
    retrieval_relevant = (
        classification_correct
        and len(matched_documents) > 0
    )

    if retrieval_relevant:
        relevant_retrievals += 1

    print("\n" + "=" * 60)
    print(f"TEST CASE {number}")
    print("=" * 60)

    print("Complaint:", complaint)
    print("Expected:", expected)
    print("Predicted:", category)
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Classification Correct:", classification_correct)

    print("Retrieved Document IDs:")
    for doc in matched_documents:
        print("-", doc["doc_id"])

    print("Category Retrieval Check:", retrieval_relevant)

print("\nFINAL TEST SUMMARY")
print(f"Correct classifications: {correct_classifications}/4")
print(f"Category-matched retrievals: {relevant_retrievals}/4")

print("\nTesting completed successfully!")