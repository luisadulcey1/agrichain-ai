
import importlib.util
from pathlib import Path
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# FULL MULTI-AGENT LANGGRAPH WORKFLOW
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def load_module(module_name, filename):
    path = BASE_DIR / "src" / filename

    spec = importlib.util.spec_from_file_location(
        module_name, path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


tools = load_module(
    "tools_module", "18_agrichain_tools.py"
)

investigator = load_module(
    "investigator_module", "20_investigator_agent.py"
)

planner = load_module(
    "planner_module", "21_planner_agent.py"
)

communicator = load_module(
    "communicator_module", "22_communicator_agent.py"
)

escalation = load_module(
    "escalation_module", "23_escalation_manager.py"
)


# ---------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------

class ComplaintState(TypedDict):
    complaint: str
    validated_category: Optional[str]
    verified_delay_days: Optional[int]
    category: str
    confidence: float
    needs_review: bool
    escalation_required: bool
    escalation_targets: list[str]
    investigation: dict[str, Any]
    plan: dict[str, Any]
    communication: dict[str, Any]
    status: str


# ---------------------------------------------------------
# AGENT 1 - ANALYZER
# ---------------------------------------------------------

def analyzer_node(state: ComplaintState):
    print("\n1. ANALYZER NODE")

    prediction = tools.classify_complaint(
        state["complaint"]
    )

    print("Category:", prediction["category"])
    print("Confidence:", prediction["confidence"])

    return {
        "category": prediction["category"],
        "confidence": prediction["confidence"],
        "needs_review": prediction["needs_review"],
        "status": "Analyzed"
    }


# ---------------------------------------------------------
# AGENT 5 - ESCALATION MANAGER
# ---------------------------------------------------------

def escalation_node(state: ComplaintState):
    print("\n2. ESCALATION MANAGER NODE")

    result = escalation.escalation_manager(
        state["complaint"],
        validated_category=state["validated_category"],
        verified_delay_days=state["verified_delay_days"]
    )

    print("Escalation targets:", result["escalation_targets"])

    return {
        "category": result["category"],
        "needs_review": result["needs_review"],
        "escalation_required": result["escalation_required"],
        "escalation_targets": result["escalation_targets"],
        "status": result["status"]
    }


def route_after_escalation(state: ComplaintState):
    if state["needs_review"]:
        return "human_review"

    # Operations escalation also requires a human decision.
    if "Operations Manager" in state["escalation_targets"]:
        return "human_review"

    return "investigate"


# ---------------------------------------------------------
# AGENT 2 - INVESTIGATOR
# ---------------------------------------------------------

def investigator_node(state: ComplaintState):
    print("\n3. INVESTIGATOR NODE")

    result = investigator.investigator_agent(
        state["complaint"],
        validated_category=state["validated_category"]
    )

    return {
        "investigation": result,
        "status": result["status"]
    }


def route_after_investigation(state: ComplaintState):
    if state["investigation"].get("status") == "Policies retrieved":
        return "plan"

    return "stop"


# ---------------------------------------------------------
# AGENT 3 - PLANNER
# ---------------------------------------------------------

def planner_node(state: ComplaintState):
    print("\n4. PLANNER NODE")

    result = planner.planner_agent(
        state["investigation"]
    )

    return {
        "plan": result,
        "status": result["status"]
    }


def route_after_planning(state: ComplaintState):
    if state["plan"].get("status") == "Plan generated":
        return "communicate"

    return "stop"


# ---------------------------------------------------------
# AGENT 4 - COMMUNICATOR
# ---------------------------------------------------------

def communicator_node(state: ComplaintState):
    print("\n5. COMMUNICATOR NODE")

    result = communicator.communicator_agent(
        state["complaint"],
        state["plan"]
    )

    return {
        "communication": result,
        "status": result["status"]
    }


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------

workflow = StateGraph(ComplaintState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("planner", planner_node)
workflow.add_node("communicator", communicator_node)

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "escalation")

workflow.add_conditional_edges(
    "escalation",
    route_after_escalation,
    {
        "human_review": END,
        "investigate": "investigator"
    }
)

workflow.add_conditional_edges(
    "investigator",
    route_after_investigation,
    {
        "plan": "planner",
        "stop": END
    }
)

workflow.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "communicate": "communicator",
        "stop": END
    }
)

workflow.add_edge("communicator", END)

app = workflow.compile()


# ---------------------------------------------------------
# TEST FULL WORKFLOW
# ---------------------------------------------------------

if __name__ == "__main__":

    complaint = (
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived. I urgently need these "
        "supplies for my farming operations."
    )

    initial_state = {
        "complaint": complaint,

        # No human validation provided
         "validated_category": None,

        # No independently verified delay information
        "verified_delay_days": None,

        "category": "",
        "confidence": 0.0,
        "needs_review": False,
        "escalation_required": False,
        "escalation_targets": [],
        "investigation": {},
        "plan": {},
        "communication": {},
        "status": "New"
    }

    print("\nAGRiCHAIN - FULL MULTI-AGENT WORKFLOW")

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL WORKFLOW REPORT")
    print("=" * 60)

    print("\nCategory:", result["category"])
    print(f'ANN Confidence: {result["confidence"]:.2%}')
    print("Needs Review:", result["needs_review"])
    print("Escalation Required:", result["escalation_required"])
    print("Escalation Targets:", result["escalation_targets"])
    print("Final Status:", result["status"])

    if result["investigation"]:
        print(
            "\nPolicies retrieved:",
            len(result["investigation"].get("policies", []))
        )

    if result["plan"].get("resolution_plan"):
        print("\nRESOLUTION PLAN:")
        print(result["plan"]["resolution_plan"])

    if result["communication"].get("customer_response"):
        print("\nCUSTOMER EMAIL DRAFT:")
        print(result["communication"]["customer_response"])

    print("\nFull workflow test completed!")
