import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import tensorflow as tf
import faiss
import joblib
import numpy as np

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# Reusable Classification and Retrieval Tools
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load existing models
print("Loading AgriChain AI models...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

ann_model = tf.keras.models.load_model(
    MODELS_DIR / "complaint_ann.keras"
)

encoder = joblib.load(
    MODELS_DIR / "category_encoder.joblib"
)

faiss_index = faiss.read_index(
    str(MODELS_DIR / "knowledge_base.index")
)

with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)


# ---------------------------------------------------------
# TOOL 1 - COMPLAINT CLASSIFICATION
# ---------------------------------------------------------

def classify_complaint(complaint_text):
    """
    Classify a customer complaint using MiniLM + ANN.

    Returns:
        category
        confidence
        needs_review
    """

    embedding = embedding_model.encode(
        [complaint_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    probabilities = ann_model.predict(
        embedding,
        verbose=0
    )[0]

    predicted_label = int(np.argmax(probabilities))

    category = encoder.inverse_transform(
        [predicted_label]
    )[0]

    confidence = float(probabilities[predicted_label])

    # Flag uncertain classifications for review
    needs_review = confidence < 0.70

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "needs_review": needs_review
    }


# ---------------------------------------------------------
# TOOL 2 - FAISS POLICY RETRIEVAL
# ---------------------------------------------------------

def retrieve_policies(complaint_text, category, top_k=3):
    """
    Retrieve category-matched policies using FAISS.
    """

    embedding = embedding_model.encode(
        [complaint_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(
        embedding,
        faiss_index.ntotal
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        doc = documents[int(idx)]

        if doc.get("category") == category:
            results.append({
                "doc_id": doc["doc_id"],
                "category": doc["category"],
                "doc_type": doc["doc_type"],
                "content": doc["content"],
                "similarity": round(float(score), 4)
            })

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# TEST REUSABLE TOOLS
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed to "
        "arrive five days ago, but the shipment "
        "has not arrived."
    )

    print("\nAGRiCHAIN - AGENT TOOLS TEST")

    # Step 1: Classify complaint
    prediction = classify_complaint(complaint)

    print("\nCLASSIFICATION:")
    print(prediction)

    # Step 2: Check classification confidence
    if prediction["needs_review"]:
        print("\nWARNING: LOW CLASSIFICATION CONFIDENCE")
        print("Manual review or additional analysis required.")
        policies = []
    else:
        # Step 3: Retrieve relevant policies
        policies = retrieve_policies(
            complaint,
            prediction["category"]
        )

    # Step 4: Display retrieved policies
    print("\nRETRIEVED POLICIES:")

    for policy in policies:
        print(
            policy["doc_id"],
            "-",
            policy["category"]
        )

    print("\nAgent tools test completed!")
