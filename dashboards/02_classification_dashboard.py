
import streamlit as st
import sys
import importlib.util
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - DASHBOARD 2
# ANN + HUMAN REVIEW + LANGGRAPH
# + EMAIL APPROVAL + SQLITE INTEGRATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgriChain AI | Complaint Classification",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TOOLS_PATH = BASE_DIR / "src" / "18_agrichain_tools.py"

WORKFLOW_PATH = (
    BASE_DIR / "src" / "25_full_langgraph_workflow.py"
)

DATABASE_PATH = (
    BASE_DIR / "src" / "26_resolution_database.py"
)

REVIEW_THRESHOLD = 0.70

COMPLAINT_CATEGORIES = [
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
# MODULE LOADER
# ---------------------------------------------------------

def import_python_module(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


@st.cache_resource
def load_classification_tools():

    return import_python_module(
        "agrichain_dashboard_tools",
        TOOLS_PATH
    )


@st.cache_resource
def load_langgraph_workflow():

    return import_python_module(
        "agrichain_dashboard_workflow",
        WORKFLOW_PATH
    )


@st.cache_resource
def load_resolution_database():

    module = import_python_module(
        "agrichain_dashboard_database",
        DATABASE_PATH
    )

    module.initialize_database()

    return module


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

DEFAULT_STATE = {
    "classification_result": None,
    "classification_number": 0,
    "review_approved": False,
    "reviewed_category": None,
    "reviewed_complaint": None,
    "review_signature": None,
    "workflow_result": None,
    "workflow_signature": None,
    "workflow_number": 0,
    "email_approved": False,
    "approved_email": None,
    "email_approval_signature": None,

    # Database integration
    "saved_complaint_id": None,
    "saved_case_signature": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# RESET FUNCTIONS
# ---------------------------------------------------------

def reset_database_save():

    st.session_state.saved_complaint_id = None
    st.session_state.saved_case_signature = None


def reset_email_approval():

    st.session_state.email_approved = False
    st.session_state.approved_email = None
    st.session_state.email_approval_signature = None

    reset_database_save()


def reset_workflow():

    st.session_state.workflow_result = None
    st.session_state.workflow_signature = None

    reset_email_approval()


def reset_review():

    st.session_state.review_approved = False
    st.session_state.reviewed_category = None
    st.session_state.reviewed_complaint = None
    st.session_state.review_signature = None

    reset_workflow()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🌱 AgriChain AI")

st.sidebar.markdown(
    "AI-Powered Complaint Resolution System"
)

st.sidebar.divider()

st.sidebar.info(
    "Dashboard 2: AI Complaint Classification"
)

st.sidebar.markdown(
    """
    **AI Components**

    - ANN Complaint Classifier
    - MiniLM Text Embeddings
    - Human Classification Review
    - LangGraph Multi-Agent Workflow
    - FAISS / RAG Policy Retrieval
    - Human Email Approval
    - SQLite Case Storage
    """
)

st.sidebar.markdown(
    "**Human Review Threshold:** 70%"
)

st.sidebar.caption(
    "AgriChain AI | Capstone Project"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🤖 AgriChain AI")

st.subheader(
    "AI-Powered Complaint Classification Dashboard"
)

st.markdown(
    """
    Classify a customer complaint using our trained
    Artificial Neural Network.

    Review the category, execute the multi-agent
    workflow, approve the customer email, and
    save the completed case to SQLite.
    """
)

st.divider()


# ---------------------------------------------------------
# CUSTOMER COMPLAINT INPUT
# ---------------------------------------------------------

st.header("📝 Customer Complaint")

complaint_text = st.text_area(
    "Enter the customer complaint:",
    height=180,
    placeholder=(
        "My agricultural supplies were supposed "
        "to arrive five days ago, but the shipment "
        "has not arrived."
    )
)

if (
    st.session_state.classification_result is not None
    and complaint_text.strip()
    != st.session_state.classification_result["complaint"]
):

    st.session_state.classification_result = None

    reset_review()


classify_button = st.button(
    "🤖 Classify Complaint",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# ANN CLASSIFICATION
# ---------------------------------------------------------

if classify_button:

    if not complaint_text.strip():

        st.warning(
            "Please enter a customer complaint first."
        )

    else:

        reset_review()

        st.session_state.classification_result = None

        st.session_state.classification_number += 1

        try:

            with st.spinner(
                "Classifying customer complaint..."
            ):

                tools = load_classification_tools()

                result = tools.classify_complaint(
                    complaint_text.strip()
                )

            st.session_state.classification_result = {
                "complaint": complaint_text.strip(),
                "category": result["category"],
                "confidence": float(result["confidence"]),
                "needs_review": bool(result["needs_review"]),
            }

        except Exception as error:

            st.error(
                "The complaint could not be classified."
            )

            st.exception(error)


# ---------------------------------------------------------
# DISPLAY CLASSIFICATION
# ---------------------------------------------------------

classification = st.session_state.classification_result

if classification is not None:

    predicted_category = classification["category"]

    confidence = float(
        classification["confidence"]
    )

    needs_review = classification["needs_review"]

    current_complaint = classification["complaint"]

    st.divider()

    st.header("📊 Classification Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Category",
            predicted_category
        )

    with col2:

        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

    st.progress(
        min(max(confidence, 0.0), 1.0)
    )

    # -----------------------------------------------------
    # HUMAN REVIEW STATUS
    # -----------------------------------------------------

    st.divider()

    st.subheader("👤 Human Review Status")

    if needs_review:

        st.warning(
            "Human review required. "
            "The ANN confidence is below 70%."
        )

    else:

        st.success(
            "ANN confidence meets the 70% threshold."
        )

    st.subheader("📋 Classification Details")

    st.json({
        "complaint": current_complaint,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "needs_review": needs_review,
        "review_threshold": REVIEW_THRESHOLD
    })

    # -----------------------------------------------------
    # HUMAN CLASSIFICATION REVIEW
    # -----------------------------------------------------

    st.divider()

    st.header("👤 Human Classification Review")

    if predicted_category not in COMPLAINT_CATEGORIES:

        st.error(
            "The predicted category is not supported."
        )

        st.stop()

    default_index = COMPLAINT_CATEGORIES.index(
        predicted_category
    )

    selected_category = st.selectbox(
        "Select the validated complaint category:",
        options=COMPLAINT_CATEGORIES,
        index=default_index,
        key=(
            "human_category_selection_"
            f"{st.session_state.classification_number}"
        )
    )

    current_signature = (
        st.session_state.classification_number,
        current_complaint,
        predicted_category,
        confidence,
        selected_category
    )

    if (
        st.session_state.review_signature
        != current_signature
    ):

        st.session_state.review_approved = False
        st.session_state.reviewed_category = None
        st.session_state.reviewed_complaint = None

        reset_workflow()

    if st.button(
        "✅ Confirm Human Review",
        type="primary",
        key="confirm_human_review"
    ):

        st.session_state.reviewed_category = (
            selected_category
        )

        st.session_state.reviewed_complaint = (
            current_complaint
        )

        st.session_state.review_signature = (
            current_signature
        )

        st.session_state.review_approved = True

        reset_workflow()

    # -----------------------------------------------------
    # HUMAN REVIEW CONFIRMED
    # -----------------------------------------------------

    if (
        st.session_state.review_approved
        and st.session_state.review_signature
        == current_signature
    ):

        st.success(
            "Human classification review completed."
        )

        st.metric(
            "Validated Category",
            st.session_state.reviewed_category
        )

        # -------------------------------------------------
        # LANGGRAPH WORKFLOW
        # -------------------------------------------------

        st.divider()

        st.header("🔗 Multi-Agent Resolution Workflow")

        st.info(
            "Running the workflow may incur OpenAI "
            "API costs. No customer email is sent."
        )

        run_workflow = st.button(
            "🚀 Run Multi-Agent Workflow",
            type="primary",
            key="run_langgraph_workflow",
            use_container_width=True
        )

        if run_workflow:

            reset_workflow()

            st.session_state.workflow_number += 1

            try:

                with st.spinner(
                    "Running AgriChain AI agents..."
                ):

                    workflow_module = (
                        load_langgraph_workflow()
                    )

                    initial_state = {
                        "complaint": (
                            st.session_state.reviewed_complaint
                        ),
                        "validated_category": (
                            st.session_state.reviewed_category
                        ),
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

                    result = workflow_module.app.invoke(
                        initial_state
                    )

                    st.session_state.workflow_result = (
                        result
                    )

                    st.session_state.workflow_signature = (
                        current_signature
                    )

                st.success(
                    "LangGraph workflow execution completed!"
                )

            except Exception as error:

                st.error(
                    "The multi-agent workflow failed."
                )

                st.exception(error)

        # -------------------------------------------------
        # DISPLAY WORKFLOW RESULTS
        # -------------------------------------------------

        workflow_result = (
            st.session_state.workflow_result
        )

        if (
            workflow_result is not None
            and st.session_state.workflow_signature
            == current_signature
        ):

            st.divider()

            st.header("📊 Multi-Agent Workflow Results")

            st.metric(
                "Final Workflow Status",
                workflow_result.get(
                    "status",
                    "Unknown"
                )
            )

            st.markdown("**Validated Category:**")
            st.write(
                st.session_state.reviewed_category
            )

            escalation_targets = (
                workflow_result.get(
                    "escalation_targets",
                    []
                )
            )

            if escalation_targets:

                st.subheader("🚨 Escalation Manager")
                st.write(escalation_targets)

            investigation = workflow_result.get(
                "investigation",
                {}
            )

            if investigation:

                st.subheader("🔎 Investigator Agent")
                st.json(investigation)

            plan = workflow_result.get(
                "plan",
                {}
            )

            if plan:

                st.subheader("🧠 Planner Agent")
                st.json(plan)

            communication = workflow_result.get(
                "communication",
                {}
            )

            if communication:

                st.subheader("✉️ Communicator Agent")
                st.json(communication)

            generated_email = communication.get(
                "customer_response",
                ""
            )

            # -------------------------------------------------
            # HUMAN EMAIL APPROVAL
            # -------------------------------------------------

            if generated_email:

                st.divider()

                st.header("✉️ Human Email Approval")

                st.warning(
                    "Verify all claims and compensation "
                    "before approving the email."
                )

                email_widget_key = (
                    "email_editor_"
                    f"{st.session_state.workflow_number}"
                )

                edited_email = st.text_area(
                    "Customer Email Draft (English)",
                    value=generated_email,
                    height=350,
                    key=email_widget_key
                )

                email_signature = (
                    st.session_state.workflow_number,
                    current_signature,
                    edited_email
                )

                if (
                    st.session_state.email_approval_signature
                    != email_signature
                ):

                    reset_email_approval()

                if st.session_state.email_approved:

                    st.success(
                        "Email Status: APPROVED"
                    )

                else:

                    st.warning(
                        "Email Status: PENDING APPROVAL"
                    )

                approve_email = st.button(
                    "✅ Approve Email Draft",
                    type="primary",
                    key="approve_email_draft",
                    disabled=not edited_email.strip()
                )

                if approve_email:

                    st.session_state.email_approved = True

                    st.session_state.approved_email = (
                        edited_email
                    )

                    st.session_state.email_approval_signature = (
                        email_signature
                    )

                    st.success(
                        "Customer email approved successfully!"
                    )

                if st.session_state.email_approved:

                    st.download_button(
                        label="📥 Download Approved Email (.txt)",
                        data=(
                            st.session_state.approved_email
                            .encode("utf-8")
                        ),
                        file_name=(
                            "agrichain_approved_customer_email.txt"
                        ),
                        mime="text/plain"
                    )

                # -------------------------------------------------
                # SQLITE DATABASE INTEGRATION
                # -------------------------------------------------

                st.divider()

                st.header("💾 Save Case to Database")

                st.markdown(
                    "Save the reviewed complaint, AI-generated "
                    "resolution plan, and approved email to "
                    "the persistent SQLite database."
                )

                if not st.session_state.email_approved:

                    st.info(
                        "Approve the customer email before "
                        "saving the completed case."
                    )

                else:

                    save_signature = (
                        st.session_state.workflow_number,
                        current_signature,
                        st.session_state.approved_email
                    )

                    already_saved = (
                        st.session_state.saved_complaint_id
                        is not None
                        and st.session_state.saved_case_signature
                        == save_signature
                    )

                    if already_saved:

                        st.success(
                            "Case saved successfully! "
                            f"Complaint ID: "
                            f"#{st.session_state.saved_complaint_id}"
                        )

                        st.caption(
                            "This case has already been saved "
                            "during the current session."
                        )

                    else:

                        save_case = st.button(
                            "💾 Save Case to Database",
                            type="primary",
                            key="save_case_to_database",
                            use_container_width=True
                        )

                        if save_case:

                            try:

                                db = load_resolution_database()

                                # Store the planner output as
                                # structured JSON text.
                                new_complaint_id = (
                                    db.create_complaint(
                                        complaint_text=current_complaint,
                                        predicted_category=predicted_category,
                                        validated_category=(
                                            st.session_state.reviewed_category
                                        ),
                                        confidence=confidence,
                                        status="Pending Approval",
                                        resolution_plan=plan,
                                        customer_email=None,
                                        email_approved=False
                                    )
                                )

                                # Save the approved draft using
                                # the existing database functions.
                                db.save_customer_email(
                                    new_complaint_id,
                                    st.session_state.approved_email
                                )

                                db.approve_customer_email(
                                    new_complaint_id
                                )

                                st.session_state.saved_complaint_id = (
                                    new_complaint_id
                                )

                                st.session_state.saved_case_signature = (
                                    save_signature
                                )

                                st.success(
                                    "Case saved successfully! "
                                    f"Complaint ID: #{new_complaint_id}"
                                )

                                st.info(
                                    "Open Dashboard 4 on port 8504 "
                                    "to view the saved complaint."
                                )

                            except Exception as error:

                                st.error(
                                    "The case could not be saved "
                                    "to SQLite."
                                )

                                st.exception(error)

            # -------------------------------------------------
            # WORKFLOW SAFETY STATUS
            # -------------------------------------------------

            st.divider()

            st.subheader("🛡️ Workflow Safety Status")

            if workflow_result.get(
                "needs_review",
                False
            ):

                st.warning(
                    "Additional human review is required."
                )

            elif workflow_result.get(
                "escalation_required",
                False
            ):

                st.warning(
                    "Additional escalation review is required."
                )

            else:

                st.success(
                    "No additional escalation flag "
                    "was returned by the workflow."
                )

else:

    st.info(
        "Enter a complaint and click "
        "'Classify Complaint' to begin."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AgriChain AI | ANN + LangGraph + Human Approval "
    "| SQLite Integration | Capstone Project"
)
