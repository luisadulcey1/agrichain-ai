
import importlib.util
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# AGENT 2: INVESTIGATOR AGENT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load reusable tools from file 18
tools_path = BASE_DIR / "src" / "18_agrichain_tools.py"

spec = importlib.util.spec_from_file_location(
    "agrichain_tools",
    tools_path
)

agrichain_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agrichain_tools)

classify_complaint = agrichain_tools.classify_complaint
retrieve_policies = agrichain_tools.retrieve_policies


# ---------------------------------------------------------
# INVESTIGATOR AGENT
# ---------------------------------------------------------

def investigator_agent(complaint_text, validated_category=None):
    """
    Investigate a customer complaint using ANN and FAISS.

    If ANN confidence is low, policy retrieval is paused
    unless a category has been explicitly validated.

    validated_category is intended for a human-reviewed
    category, not an automatic confidence override.
    """

    print("\nINVESTIGATOR AGENT STARTED")

    # Step 1: Classify complaint using ANN
    prediction = classify_complaint(complaint_text)

    predicted_category = prediction["category"]
    confidence = prediction["confidence"]
    needs_review = prediction["needs_review"]

    category = predicted_category
    human_validated = False

    print("\nANN CLASSIFICATION:")
    print("Predicted category:", predicted_category)
    print(f"Confidence: {confidence:.2%}")
    print("Needs review:", needs_review)

    # Step 2: Apply explicit human validation if provided
    if validated_category is not None:

        valid_categories = set(
            agrichain_tools.encoder.classes_
        )

        if validated_category not in valid_categories:
            raise ValueError(
                f"Invalid validated category: {validated_category}"
            )

        category = validated_category
        needs_review = False
        human_validated = True

        print("\nHUMAN VALIDATION (SIMULATED):")
        print("Confirmed category:", category)

    # Step 3: Stop retrieval if classification is uncertain
    if needs_review:

        print("\nClassification requires human review.")
        print("Policy retrieval paused.")

        return {
            "complaint": complaint_text,
            "predicted_category": predicted_category,
            "category": category,
            "confidence": confidence,
            "needs_review": True,
            "human_validated": False,
            "policies": [],
            "status": "Awaiting category validation"
        }

    # Step 4: Retrieve matching policies from FAISS
    print("\nSearching FAISS knowledge base...")

    policies = retrieve_policies(
        complaint_text,
        category,
        top_k=3
    )

    status = (
        "Policies retrieved"
        if policies
        else "No matching policies found"
    )

    # Step 5: Return structured investigation report
    return {
        "complaint": complaint_text,
        "predicted_category": predicted_category,
        "category": category,
        "confidence": confidence,
        "needs_review": False,
        "human_validated": human_validated,
        "policies": policies,
        "status": status
    }


# ---------------------------------------------------------
# TEST INVESTIGATOR AGENT
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    # Simulate a human confirming the category
    result = investigator_agent(
        complaint,
        validated_category="Delivery Delay"
    )

    print("\n" + "=" * 60)
    print("INVESTIGATOR AGENT REPORT")
    print("=" * 60)

    print("\nComplaint:")
    print(result["complaint"])

    print("\nANN Predicted Category:")
    print(result["predicted_category"])

    print("\nFinal Category:")
    print(result["category"])

    print("\nANN Confidence:")
    print(f'{result["confidence"]:.2%}')

    print("\nNeeds Review:")
    print(result["needs_review"])

    print("\nHuman Validated (Simulated):")
    print(result["human_validated"])

    print("\nInvestigation Status:")
    print(result["status"])

    print("\nRETRIEVED POLICIES:")

    if result["policies"]:

        for policy in result["policies"]:
            print(
                f"\nDocument ID: {policy['doc_id']}"
            )
            print(
                f"Category: {policy['category']}"
            )
            print(
                f"Similarity: {policy['similarity']}"
            )
            print(
                f"Policy: {policy['content']}"
            )

    else:
        print("No policies retrieved.")

    print("\nInvestigator Agent completed successfully!")
