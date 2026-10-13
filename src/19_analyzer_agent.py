
import importlib.util
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# AGENT 1: ANALYZER AGENT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# Import classification tools from file 18
tools_path = BASE_DIR / "src" / "18_agrichain_tools.py"

spec = importlib.util.spec_from_file_location(
    "agrichain_tools",
    tools_path
)

agrichain_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agrichain_tools)

classify_complaint = agrichain_tools.classify_complaint

# Initialize OpenAI
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def analyzer_agent(complaint_text):
    """
    Analyze complaint category, severity,
    and potential escalation requirements.
    """

    print("\nANALYZER AGENT STARTED")

    # Step 1: ANN classification
    prediction = classify_complaint(complaint_text)

    category = prediction["category"]
    confidence = prediction["confidence"]
    needs_review = prediction["needs_review"]

    print("\nANN CLASSIFICATION:")
    print("Category:", category)
    print("Confidence:", confidence)
    print("Needs review:", needs_review)

    # Step 2: LLM severity analysis
    system_prompt = """
You are the Analyzer Agent for AgriChain AI.

Your responsibility is to analyze agricultural
supply chain customer complaints.

Determine:
1. Complaint severity: Low, Medium, High, or Critical.
2. The reasons for the severity assessment.
3. Whether escalation should be considered.
4. Whether the ANN category appears consistent
   with the complaint.

Important rules:
- Do not invent facts.
- Do not claim an investigation has been completed.
- Do not change the ANN prediction silently.
- If classification confidence is low, recommend
  human review.
- A severity assessment is preliminary and must
  be confirmed against company escalation policies.
- Respond in professional English.
"""

    user_prompt = f"""
CUSTOMER COMPLAINT:
{complaint_text}

ANN PREDICTED CATEGORY:
{category}

ANN CONFIDENCE:
{confidence:.2%}

CLASSIFICATION NEEDS REVIEW:
{needs_review}

Analyze this complaint and explain your assessment.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    result = {
        "complaint": complaint_text,
        "predicted_category": category,
        "confidence": confidence,
        "needs_review": needs_review,
        "severity_analysis": response.content
    }

    return result


# ---------------------------------------------------------
# TEST ANALYZER AGENT
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    result = analyzer_agent(complaint)

    print("\n" + "=" * 60)
    print("ANALYZER AGENT REPORT")
    print("=" * 60)

    print("\nComplaint:")
    print(result["complaint"])

    print("\nPredicted Category:")
    print(result["predicted_category"])

    print("\nConfidence:")
    print(f'{result["confidence"]:.2%}')

    print("\nNeeds Review:")
    print(result["needs_review"])

    print("\nSeverity Analysis:")
    print(result["severity_analysis"])

    print("\nAnalyzer Agent completed successfully!")
