
import os
import importlib.util
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# AGENT 3: PLANNER AGENT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Import Investigator Agent
investigator_path = (
    BASE_DIR / "src" / "20_investigator_agent.py"
)

spec = importlib.util.spec_from_file_location(
    "investigator_module",
    investigator_path
)

investigator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(investigator_module)

investigator_agent = investigator_module.investigator_agent

# Initialize OpenAI
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0
)


def planner_agent(investigation):
    """
    Generate a resolution plan from retrieved policies.
    """

    print("\nPLANNER AGENT STARTED")

    if investigation["status"] != "Policies retrieved":
        return {
            "status": "Blocked",
            "resolution_plan": None,
            "reason": "No validated policy context available."
        }

    policies = investigation["policies"]

    if not policies:
        return {
            "status": "Blocked",
            "resolution_plan": None,
            "reason": "No policies were retrieved."
        }

    policy_context = "\n\n".join(
        f"Document ID: {policy['doc_id']}\n"
        f"Policy: {policy['content']}"
        for policy in policies
    )

    system_prompt = """
You are the Planner Agent for AgriChain AI.

Your job is to prepare an actionable complaint
resolution plan using retrieved company policies.

Rules:
1. Use only the provided policies.
2. Do not invent tracking results, ETAs or actions taken.
3. Clearly distinguish proposed actions from
   completed actions.
4. Do not promise compensation before eligibility
   and required approvals are verified.
5. Cite the document IDs supporting each action.
6. Identify missing information.
7. Explain when escalation may be required.
8. Respond in professional English.

Use these sections:

Resolution Objective
Investigation Steps
Proposed Resolution
Compensation Eligibility
Escalation Conditions
Missing Information
Policy References
"""

    user_prompt = f"""
CUSTOMER COMPLAINT:
{investigation['complaint']}

VALIDATED CATEGORY:
{investigation['category']}

RETRIEVED COMPANY POLICIES:
{policy_context}

Prepare a grounded resolution plan.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "status": "Plan generated",
        "category": investigation["category"],
        "source_document_ids": [
            policy["doc_id"] for policy in policies
        ],
        "resolution_plan": response.content
    }


# ---------------------------------------------------------
# TEST PLANNER AGENT
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    # Simulated human category validation
    investigation = investigator_agent(
        complaint,
        validated_category="Delivery Delay"
    )

    result = planner_agent(investigation)

    print("\n" + "=" * 60)
    print("PLANNER AGENT REPORT")
    print("=" * 60)

    print("\nStatus:", result["status"])

    if result["resolution_plan"]:
        print("\nRESOLUTION PLAN:")
        print(result["resolution_plan"])
    else:
        print("\nReason:", result["reason"])

    print("\nPlanner Agent completed!")
