
import streamlit as st
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - DASHBOARD 3
# MULTI-AGENT WORKFLOW MONITORING
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgriChain AI | Workflow Monitor",
    page_icon="🔗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

WORKFLOW_PATH = (
    BASE_DIR / "src" / "25_full_langgraph_workflow.py"
)

CATEGORIES = [
    "Damaged Goods",
    "Delivery Delay",
    "Documentation Error",
    "Missing Items",
    "Poor Communication",
    "Pricing Error",
    "Quality Issues",
    "Service Complaint",
]


# ---------------------------------------------------------
# LOAD LANGGRAPH
# ---------------------------------------------------------

@st.cache_resource
def load_workflow():

    spec = importlib.util.spec_from_file_location(
        "agrichain_monitor_workflow",
        WORKFLOW_PATH
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load LangGraph workflow.")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module.app


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "monitor_result" not in st.session_state:
    st.session_state.monitor_result = None

if "monitor_signature" not in st.session_state:
    st.session_state.monitor_signature = None

if "monitor_approved" not in st.session_state:
    st.session_state.monitor_approved = False

if "monitor_approval_signature" not in st.session_state:
    st.session_state.monitor_approval_signature = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🌱 AgriChain AI")
st.sidebar.markdown("AI-Powered Complaint Resolution System")
st.sidebar.divider()
st.sidebar.info("Dashboard 3: Multi-Agent Workflow Monitor")

st.sidebar.markdown(
    """
    **Agents**

    1. Analyzer
    2. Escalation Manager
    3. Investigator
    4. Planner
    5. Communicator
    """
)

st.sidebar.caption("AgriChain AI | Capstone Project")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🔗 Multi-Agent Workflow Monitor")

st.markdown(
    "Monitor the results of the AgriChain AI "
    "LangGraph complaint resolution workflow."
)

st.info(
    "This dashboard displays workflow results after "
    "execution. It does not stream live agent activity."
)

st.divider()


# ---------------------------------------------------------
# COMPLAINT INPUT
# ---------------------------------------------------------

st.header("📝 Complaint Input")

complaint = st.text_area(
    "Customer Complaint",
    height=140,
    placeholder=(
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived."
    )
)

validated_category = st.selectbox(
    "Human-Validated Category",
    options=CATEGORIES
)

st.warning(
    "Select the correct category based on a human "
    "review of the complaint. The selection alone "
    "does not count as approval."
)

current_signature = (
    complaint.strip(),
    validated_category
)

# Invalidate approvals and old results when inputs change.
if (
    st.session_state.monitor_approval_signature
    != current_signature
):
    st.session_state.monitor_approved = False

if st.session_state.monitor_signature != current_signature:
    st.session_state.monitor_result = None

confirm = st.button(
    "✅ Confirm Human Classification",
    disabled=not complaint.strip()
)

if confirm:
    st.session_state.monitor_approved = True
    st.session_state.monitor_approval_signature = (
        current_signature
    )
    st.success("Human classification confirmed.")

if st.session_state.monitor_approved:
    st.success(
        f"Validated category: {validated_category}"
    )

st.divider()


# ---------------------------------------------------------
# EXECUTE WORKFLOW
# ---------------------------------------------------------

st.header("🚀 Workflow Execution")

st.caption(
    "Execution may use OpenAI API calls and incur costs. "
    "No customer email will be sent."
)

run = st.button(
    "Run Multi-Agent Workflow",
    type="primary",
    use_container_width=True,
    disabled=not st.session_state.monitor_approved
)

if run:

    st.session_state.monitor_result = None

    try:

        with st.spinner("Executing LangGraph agents..."):

            app = load_workflow()

            initial_state = {
                "complaint": complaint.strip(),
                "validated_category": validated_category,
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

            result = app.invoke(initial_state)

            st.session_state.monitor_result = result
            st.session_state.monitor_signature = (
                current_signature
            )

        st.success("Workflow execution completed.")

    except Exception as error:
        st.error("Workflow execution failed.")
        st.exception(error)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

result = st.session_state.monitor_result

if (
    result is not None
    and st.session_state.monitor_signature
    == current_signature
):

    st.divider()
    st.header("📊 Workflow Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Final Status",
            result.get("status", "Unknown")
        )

    with col2:
        st.metric(
            "ANN Confidence",
            f"{float(result.get('confidence', 0)) * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Final Category",
            result.get("category", "Unknown")
        )

    st.divider()

    # -----------------------------------------------------
    # AGENT STATUS
    # -----------------------------------------------------

    st.header("🤖 Agent Execution Summary")

    investigation = result.get("investigation", {})
    plan = result.get("plan", {})
    communication = result.get("communication", {})

    agent_statuses = {
        "Analyzer": (
            "Completed"
            if result.get("category")
            else "Not completed"
        ),
        "Escalation Manager": (
            "Completed"
            if result.get("status") != "New"
            else "Not confirmed"
        ),
        "Investigator": (
            "Completed"
            if investigation
            else "Not executed"
        ),
        "Planner": (
            "Completed"
            if plan
            else "Not executed"
        ),
        "Communicator": (
            "Completed"
            if communication
            else "Not executed"
        ),
    }

    for agent_name, status in agent_statuses.items():

        col_a, col_b = st.columns([3, 1])

        with col_a:
            st.write(f"**{agent_name}**")

        with col_b:
            if status == "Completed":
                st.success(status)
            else:
                st.info(status)

    # -----------------------------------------------------
    # ESCALATION DETAILS
    # -----------------------------------------------------

    st.divider()
    st.header("🚨 Escalation Manager")

    st.write(
        "**Human Review Required:**",
        result.get("needs_review", False)
    )

    st.write(
        "**Escalation Required:**",
        result.get("escalation_required", False)
    )

    st.write(
        "**Escalation Targets:**",
        result.get("escalation_targets", [])
    )

    # -----------------------------------------------------
    # INVESTIGATOR DETAILS
    # -----------------------------------------------------

    st.divider()
    st.header("🔎 Investigator Agent")

    if investigation:

        policies = investigation.get("policies", [])

        st.metric(
            "Policies Retrieved",
            len(policies)
        )

        for policy in policies:

            with st.expander(
                f"{policy.get('doc_id', 'Policy')} "
                f"— {policy.get('category', '')}"
            ):
                st.write(policy.get("content", ""))

                st.caption(
                    "Similarity: "
                    f"{policy.get('similarity', 0):.4f}"
                )

    else:
        st.info("Investigator was not executed.")

    # -----------------------------------------------------
    # PLANNER DETAILS
    # -----------------------------------------------------

    st.divider()
    st.header("🧠 Planner Agent")

    if plan:
        st.markdown(
            plan.get(
                "resolution_plan",
                "No resolution plan available."
            )
        )
    else:
        st.info("Planner was not executed.")

    # -----------------------------------------------------
    # COMMUNICATOR DETAILS
    # -----------------------------------------------------

    st.divider()
    st.header("✉️ Communicator Agent")

    if communication:

        customer_response = communication.get(
            "customer_response", ""
        )

        if customer_response:

            st.text_area(
                "Generated Customer Email",
                value=customer_response,
                height=300,
                disabled=True
            )

            st.warning(
                "Draft only. Human approval is required "
                "before sending."
            )

        else:
            st.info("No customer response was generated.")

    else:
        st.info("Communicator was not executed.")

    # -----------------------------------------------------
    # RAW STATE
    # -----------------------------------------------------

    st.divider()

    with st.expander("🔧 View Complete LangGraph State"):
        st.json(result)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AgriChain AI | Multi-Agent Workflow Monitoring "
    "| Dashboard 3 | Capstone Project"
)
