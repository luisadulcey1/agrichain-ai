
import importlib.util
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 3
# LANGGRAPH WORKFLOW WITH REAL AGENTS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def load_module(module_name, filename):
    module_path = BASE_DIR / "src" / filename

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# Import existing tools and agents
tools_module = load_module(
    "agrichain_tools",
    "18_agrichain_tools.py"
)

escalation_module = load_module(
    "escalation_module",
    "23_escalation_manager.py"
)

classify_complaint = tools_module.classify_complaint
escalation_manager = escalation_module.escalation_manager


# ---------------------------------------------------------
# SHARED WORKFLOW STATE
# ---------------------------------------------------------

class ComplaintState(TypedDict):
    complaint: str
    category: str
    confidence: float
    needs_review: bool
    escalation_required: bool
    escalation_targets: list[str]
    status: str


# ---------------------------------------------------------
# NODE 1 - ANALYZER
# ---------------------------------------------------------

def analyzer_node(state: ComplaintState):
    print("\n1. ANALYZER NODE")

    prediction = classify_complaint(
        state["complaint"]
    )

    print("Category:", prediction["category"])
    print("Confidence:", prediction["confidence"])
    print("Needs review:", prediction["needs_review"])

    return {
        "category": prediction["category"],
        "confidence": prediction["confidence"],
        "needs_review": prediction["needs_review"],
        "status": "Analyzed"
    }


# ---------------------------------------------------------
# NODE 2 - ESCALATION MANAGER
# ---------------------------------------------------------

def escalation_node(state: ComplaintState):
    print("\n2. ESCALATION MANAGER NODE")

    result = escalation_manager(
        state["complaint"]
    )

    print("Escalation required:", result["escalation_required"])
    print("Targets:", result["escalation_targets"])

    return {
        "escalation_required": result["escalation_required"],
        "escalation_targets": result["escalation_targets"],
        "needs_review": result["needs_review"],
        "status": result["status"]
    }


# ---------------------------------------------------------
# ROUTING LOGIC
# ---------------------------------------------------------

def route_complaint(state: ComplaintState):

    if state["needs_review"]:
        return "review"

    return "continue"


# ---------------------------------------------------------
# NODE 3 - INVESTIGATOR PLACEHOLDER
# ---------------------------------------------------------

def investigator_node(state: ComplaintState):
    print("\n3. INVESTIGATOR NODE")

    print("Complaint is ready for policy retrieval.")

    return {
        "status": "Ready for policy retrieval"
    }


# ---------------------------------------------------------
# BUILD LANGGRAPH
# ---------------------------------------------------------

workflow = StateGraph(ComplaintState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("investigator", investigator_node)

workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "escalation")

workflow.add_conditional_edges(
    "escalation",
    route_complaint,
    {
        "review": END,
        "continue": "investigator"
    }
)

workflow.add_edge("investigator", END)

app = workflow.compile()


# ---------------------------------------------------------
# TEST REAL WORKFLOW
# ---------------------------------------------------------

if __name__ == "__main__":

    initial_state = {
        "complaint": (
            "My agricultural supplies were supposed "
            "to arrive five days ago, but the shipment "
            "has not arrived. I urgently need these "
            "supplies for my farming operations."
        ),
        "category": "",
        "confidence": 0.0,
        "needs_review": False,
        "escalation_required": False,
        "escalation_targets": [],
        "status": "New"
    }

    print("\nAGRiCHAIN - REAL LANGGRAPH WORKFLOW")

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL WORKFLOW STATE")
    print("=" * 60)

    print("Category:", result["category"])
    print(f'Confidence: {result["confidence"]:.2%}')
    print("Needs Review:", result["needs_review"])
    print("Escalation Required:", result["escalation_required"])
    print("Escalation Targets:", result["escalation_targets"])
    print("Status:", result["status"])

    print("\nLangGraph workflow completed!")
