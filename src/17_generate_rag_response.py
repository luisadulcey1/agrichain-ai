import os
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------
# AgriChain AI - RAG Response Generation
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load ANN classifier
ann_model = tf.keras.models.load_model(
    MODELS_DIR / "complaint_ann.keras"
)

encoder = joblib.load(
    MODELS_DIR / "category_encoder.joblib"
)

# Load FAISS index
index = faiss.read_index(
    str(MODELS_DIR / "knowledge_base.index")
)

# Load knowledge base
with open(
    MODELS_DIR / "knowledge_base_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

# Initialize OpenAI
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0
)

# Customer complaint
complaint = (
    "My agricultural supplies were supposed to arrive "
    "five days ago, but the shipment has not arrived. "
    "I need an updated delivery date and a solution."
)

print("\nAGRiCHAIN - AI RAG RESOLUTION")

# Generate complaint embedding
embedding = embedding_model.encode(
    [complaint],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

# Predict complaint category
probabilities = ann_model.predict(
    embedding,
    verbose=0
)[0]

predicted_label = int(np.argmax(probabilities))

category = encoder.inverse_transform(
    [predicted_label]
)[0]

confidence = float(probabilities[predicted_label])

print("\nPREDICTED CATEGORY:", category)
print(f"CONFIDENCE: {confidence * 100:.2f}%")

# Retrieve category-matched policies
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
        retrieved_docs.append(doc)

    if len(retrieved_docs) == 3:
        break

if not retrieved_docs:
    raise ValueError(
        "No category-matched policies were found."
    )

# Prepare policy context
policy_context = "\n\n".join(
    f"Document ID: {doc['doc_id']}\n"
    f"Policy: {doc['content']}"
    for doc in retrieved_docs
)

# Generate grounded resolution
system_prompt = """
You are AgriChain AI, an agricultural supply chain
complaint resolution assistant.

Use only the retrieved policies as the basis for
your recommended resolution.

Rules:
1. Do not invent policies or compensation amounts.
2. Do not claim that an investigation has been completed.
3. Do not invent shipment tracking information.
4. Explain the recommended investigation steps.
5. Mention compensation only as a possibility
   subject to policy eligibility and approval.
6. Cite relevant document IDs such as [KB-00001].
7. If information is missing, identify what must
   be verified before taking action.
8. Write the response in professional English.

Structure your answer with these sections:

Complaint Summary
Recommended Investigation
Proposed Resolution
Customer Communication
Policy References
"""

user_prompt = f"""
CUSTOMER COMPLAINT:
{complaint}

PREDICTED CATEGORY:
{category}

RETRIEVED POLICIES:
{policy_context}

Generate a grounded complaint resolution recommendation.
"""

print("\nGENERATING AI RESPONSE...\n")

response = llm.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
])

print("=" * 60)
print("AI RESOLUTION RECOMMENDATION")
print("=" * 60)

print(response.content)

# Save report
report = {
    "complaint": complaint,
    "predicted_category": category,
    "confidence": round(confidence, 4),
    "source_document_ids": [
        doc["doc_id"] for doc in retrieved_docs
    ],
    "ai_response": response.content
}

output_path = (
    PROCESSED_DIR / "ai_rag_resolution.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nAI RAG report saved successfully!")