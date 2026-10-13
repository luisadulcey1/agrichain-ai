
import os
import importlib.util
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# AGENT 4: COMMUNICATOR AGENT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# LOAD EXISTING AGENTS
# ---------------------------------------------------------

def load_module(module_name, filename):
    """
    Load an existing Python module from the src folder.
    """

    module_path = BASE_DIR / "src" / filename

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load module: {filename}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# Import Investigator Agent
investigator_module = load_module(
    "investigator_module",
    "20_investigator_agent.py"
)

# Import Planner Agent
planner_module = load_module(
    "planner_module",
    "21_planner_agent.py"
)

investigator_agent = investigator_module.investigator_agent
planner_agent = planner_module.planner_agent


# ---------------------------------------------------------
# INITIALIZE OPENAI
# ---------------------------------------------------------

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0
)


# ---------------------------------------------------------
# AGENT 4 - COMMUNICATOR
# ---------------------------------------------------------

def communicator_agent(complaint_text, plan_result):
    """
    Generate a professional customer-facing email
    using a proposed resolution plan.

    The output is a draft pending human approval.
    """

    print("\nCOMMUNICATOR AGENT STARTED")

    # Step 1: Verify that a resolution plan exists
    if plan_result.get("status") != "Plan generated":

        return {
            "status": "Blocked",
            "customer_response": None,
            "reason": (
                "No resolution plan is available."
            )
        }

    resolution_plan = plan_result.get("resolution_plan")

    if not resolution_plan:

        return {
            "status": "Blocked",
            "customer_response": None,
            "reason": (
                "Resolution plan is empty."
            )
        }

    # -----------------------------------------------------
    # SYSTEM PROMPT - CUSTOMER COMMUNICATION RULES
    # -----------------------------------------------------

    system_prompt = """
You are the Communicator Agent for AgriChain AI.

Your responsibility is to write professional,
empathetic, and accurate customer service emails.

You receive a PROPOSED resolution plan.
The plan does not mean that any actions have occurred.

IMPORTANT FACTS:

- No shipment investigation has been completed.
- No carrier has been contacted.
- No tracking information has been verified.
- No estimated delivery date has been confirmed.
- No compensation has been approved.
- No replacement shipment has been authorized.

STRICT COMMUNICATION RULES:

1. Acknowledge the customer's complaint with empathy.

2. Never claim that an investigation has already
   started or been completed.

3. Never claim that the logistics partner or
   carrier has already been contacted.

4. Never claim that shipment tracking information
   has already been verified.

5. Never invent shipment locations, tracking
   updates, delivery dates, or ETAs.

6. Never promise discounts, refunds, service
   credits, or replacement shipping.

7. Use future-tense language when discussing
   proposed investigation steps.

8. Do not expose internal AI classifications,
   confidence scores, technical processes,
   or policy document IDs.

9. Do not mention internal company procedures
   unless relevant to the customer.

10. Keep the email concise, clear, professional,
    and empathetic.

11. Do not imply that the customer has already
    accepted a proposed resolution.

12. The email must remain a DRAFT pending
    human review and approval.

PREFERRED WORDING:

- "We will review the status of your shipment."

- "We will contact our logistics team to obtain
   additional information."

- "We will provide an update once more
   information becomes available."

- "We understand the importance of receiving
   these supplies on time."

AVOID THESE STATEMENTS:

- "We are currently investigating..."

- "We have contacted the carrier..."

- "Our team is in contact with the carrier..."

- "We have confirmed your delivery date..."

- "Your compensation has been approved..."

- "Your replacement shipment is on the way..."

OUTPUT FORMAT:

Subject:
Greeting:
Message:
Professional Closing:

Write only the customer-facing email
in professional English.
"""

    # -----------------------------------------------------
    # USER PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
CUSTOMER COMPLAINT:

{complaint_text}

PROPOSED RESOLUTION PLAN:

{resolution_plan}

Generate a professional customer-facing email.

Remember:

- The investigation has NOT started.
- No carrier has been contacted.
- No delivery date has been confirmed.
- No compensation has been approved.
- The email is a draft for human approval.
"""

    # -----------------------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------------------

    print("\nGenerating customer email draft...")

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    customer_email = response.content

    # -----------------------------------------------------
    # RETURN STRUCTURED RESULT
    # -----------------------------------------------------

    return {
        "status": "Draft generated - pending approval",
        "customer_response": customer_email,
        "requires_human_approval": True
    }


# ---------------------------------------------------------
# TEST COMMUNICATOR AGENT
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    print("\nAGRiCHAIN - COMMUNICATOR AGENT TEST")

    # Step 1: Simulate human category validation
    investigation = investigator_agent(
        complaint,
        validated_category="Delivery Delay"
    )

    # Step 2: Generate resolution plan
    plan = planner_agent(investigation)

    # Step 3: Generate customer email draft
    result = communicator_agent(
        complaint,
        plan
    )

    # -----------------------------------------------------
    # DISPLAY FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("COMMUNICATOR AGENT REPORT")
    print("=" * 60)

    print("\nStatus:")
    print(result["status"])

    print("\nRequires Human Approval:")
    print(result.get("requires_human_approval", True))

    if result["customer_response"]:

        print("\nCUSTOMER EMAIL DRAFT:\n")
        print(result["customer_response"])

    else:

        print("\nReason:")
        print(result.get("reason", "Unknown error"))

    print("\nCommunicator Agent completed!")
