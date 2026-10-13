
import streamlit as st
import pandas as pd
import importlib.util
import sys
import json
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - DASHBOARD 4
# RESOLUTION MANAGEMENT
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgriChain AI | Resolution Management",
    page_icon="📋",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_MODULE_PATH = BASE_DIR / "src" / "26_resolution_database.py"

CATEGORIES = [
    "Damaged Goods",
    "Delivery Delay",
    "Documentation Error",
    "Missing Items",
    "Poor Communication",
    "Pricing Error",
    "Quality Issues",
    "Service Complaint"
]

STATUSES = [
    "Pending Review",
    "In Progress",
    "Pending Approval",
    "Approved",
    "Resolved",
    "Escalated",
    "Closed"
]


# ---------------------------------------------------------
# LOAD DATABASE MODULE
# ---------------------------------------------------------

@st.cache_resource
def load_database():
    spec = importlib.util.spec_from_file_location(
        "agrichain_resolution_database",
        DATABASE_MODULE_PATH
    )

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load database module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.initialize_database()

    return module


db = load_database()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "selected_complaint_id" not in st.session_state:
    st.session_state.selected_complaint_id = None

if "record_version" not in st.session_state:
    st.session_state.record_version = 0


def refresh_records():
    st.session_state.record_version += 1


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🌱 AgriChain AI")
st.sidebar.markdown("AI-Powered Complaint Resolution System")
st.sidebar.divider()
st.sidebar.info("Dashboard 4: Resolution Management")

st.sidebar.markdown(
    """
    **Features**

    - Persistent SQLite Storage
    - Complaint Registry
    - Resolution Status Tracking
    - Resolution Plan Management
    - Human Email Approval
    - CSV Export
    """
)

st.sidebar.caption(
    "Local capstone prototype — no automatic email sending."
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📋 Resolution Management Dashboard")

st.markdown(
    "Manage complaint records, resolution plans, "
    "customer email drafts, and human approvals."
)

st.info(
    "Records in this dashboard are saved in SQLite. "
    "Existing results from Dashboards 2 and 3 are not "
    "automatically imported."
)

st.divider()


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

statistics = db.get_database_statistics()

st.header("📊 Resolution Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Complaints",
        statistics["total_complaints"]
    )

with col2:
    st.metric(
        "Resolved",
        statistics["resolved_complaints"]
    )

with col3:
    st.metric(
        "Escalated",
        statistics["escalated_complaints"]
    )

with col4:
    st.metric(
        "Approved Emails",
        statistics["approved_emails"]
    )

st.divider()


# ---------------------------------------------------------
# CREATE NEW COMPLAINT
# ---------------------------------------------------------

st.header("➕ Register New Complaint")

with st.form("new_complaint_form", clear_on_submit=True):

    new_complaint = st.text_area(
        "Customer Complaint",
        height=130,
        placeholder="Enter the customer complaint..."
    )

    new_category = st.selectbox(
        "Human-Validated Category",
        options=["Not Yet Validated"] + CATEGORIES
    )

    new_status = st.selectbox(
        "Initial Status",
        options=["Pending Review", "In Progress"]
    )

    submit_new = st.form_submit_button(
        "💾 Save Complaint",
        type="primary"
    )

    if submit_new:

        if not new_complaint.strip():

            st.warning(
                "Please enter a complaint before saving."
            )

        else:

            try:

                validated_category = (
                    None
                    if new_category == "Not Yet Validated"
                    else new_category
                )

                complaint_id = db.create_complaint(
                    complaint_text=new_complaint,
                    validated_category=validated_category,
                    status=new_status
                )

                st.success(
                    f"Complaint #{complaint_id} saved successfully."
                )

                refresh_records()

            except Exception as error:
                st.error("Unable to save complaint.")
                st.exception(error)

st.divider()


# ---------------------------------------------------------
# COMPLAINT REGISTRY
# ---------------------------------------------------------

st.header("🗂️ Complaint Registry")

records = db.get_all_complaints()

if records:

    dataframe = pd.DataFrame(records)

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        status_filter = st.selectbox(
            "Filter by Status",
            ["All Statuses"] + STATUSES
        )

    with filter_col2:

        category_filter = st.selectbox(
            "Filter by Category",
            ["All Categories"] + CATEGORIES
        )

    filtered_df = dataframe.copy()

    if status_filter != "All Statuses":

        filtered_df = filtered_df[
            filtered_df["status"] == status_filter
        ]

    if category_filter != "All Categories":

        filtered_df = filtered_df[
            filtered_df["validated_category"]
            == category_filter
        ]

    display_columns = [
        "id",
        "complaint_text",
        "validated_category",
        "confidence",
        "status",
        "email_approved",
        "created_at",
        "updated_at"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_df.to_csv(index=False)

    st.download_button(
        "📥 Export Filtered Complaints (CSV)",
        data=csv_data,
        file_name="agrichain_complaints.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No complaints have been saved yet. "
        "Register your first complaint above."
    )

st.divider()


# ---------------------------------------------------------
# MANAGE EXISTING COMPLAINT
# ---------------------------------------------------------

st.header("🛠️ Manage Complaint Resolution")

if records:

    complaint_ids = [record["id"] for record in records]

    selected_id = st.selectbox(
        "Select Complaint ID",
        options=complaint_ids,
        format_func=lambda complaint_id: (
            f"Complaint #{complaint_id}"
        )
    )

    record = db.get_complaint_by_id(selected_id)

    if record:

        st.subheader(f"Complaint #{selected_id}")

        st.markdown("**Customer Complaint**")
        st.write(record["complaint_text"])

        st.markdown("**Validated Category**")
        st.write(
            record["validated_category"] or "Not validated"
        )

        st.markdown("**ANN Confidence**")

        if record["confidence"] is None:
            st.write("Not available")
        else:
            st.write(
                f"{record['confidence'] * 100:.2f}%"
            )

        st.markdown("**Created At (UTC)**")
        st.write(record["created_at"])

        st.markdown("**Updated At (UTC)**")
        st.write(record["updated_at"])

        st.divider()

        # -------------------------------------------------
        # STATUS MANAGEMENT
        # -------------------------------------------------

        st.subheader("📌 Resolution Status")

        current_status = record["status"]

        new_status = st.selectbox(
            "Select Resolution Status",
            options=STATUSES,
            index=STATUSES.index(current_status),
            key=f"status_{selected_id}_{st.session_state.record_version}"
        )

        if st.button(
            "💾 Update Status",
            key=f"update_status_{selected_id}"
        ):

            if (
                new_status == "Approved"
                and not record["email_approved"]
            ):
                st.error(
                    "The email must be approved through "
                    "the Human Email Approval section."
                )

            else:

                updated = db.update_complaint_status(
                    selected_id,
                    new_status
                )

                if updated:
                    st.success("Status updated successfully.")
                    refresh_records()
                    st.rerun()

        # -------------------------------------------------
        # RESOLUTION PLAN
        # -------------------------------------------------

        st.divider()

        st.subheader("🧠 Resolution Plan")

        saved_plan = record["resolution_plan"] or ""

        edited_plan = st.text_area(
            "Resolution Plan",
            value=saved_plan,
            height=240,
            key=f"plan_{selected_id}_{st.session_state.record_version}"
        )

        if st.button(
            "💾 Save Resolution Plan",
            key=f"save_plan_{selected_id}"
        ):

            updated = db.update_resolution_plan(
                selected_id,
                edited_plan
            )

            if updated:
                st.success("Resolution plan saved.")
                refresh_records()
                st.rerun()

        # -------------------------------------------------
        # CUSTOMER EMAIL DRAFT
        # -------------------------------------------------

        st.divider()

        st.subheader("✉️ Customer Email Draft")

        saved_email = record["customer_email"] or ""

        edited_email = st.text_area(
            "English Customer Email",
            value=saved_email,
            height=300,
            key=f"email_{selected_id}_{st.session_state.record_version}"
        )

        if (
            record["email_approved"]
            and edited_email != saved_email
        ):
            st.warning(
                "The draft has been edited. Save the "
                "changes to invalidate its previous approval."
            )

        if st.button(
            "💾 Save Email Draft",
            key=f"save_email_{selected_id}"
        ):

            if not edited_email.strip():
                st.warning("Email draft cannot be empty.")

            else:

                updated = db.save_customer_email(
                    selected_id,
                    edited_email
                )

                if updated:
                    st.success(
                        "Email draft saved. "
                        "Human approval is now required."
                    )
                    refresh_records()
                    st.rerun()

        # -------------------------------------------------
        # HUMAN EMAIL APPROVAL
        # -------------------------------------------------

        st.divider()

        st.subheader("👤 Human Email Approval")

        if record["email_approved"]:

            st.success("Email Status: APPROVED")

        elif record["customer_email"]:

            st.warning("Email Status: PENDING APPROVAL")

        else:

            st.info("No email draft has been saved yet.")

        # Prevent approval of unsaved edits.
        has_unsaved_email_changes = (
            edited_email != saved_email
        )

        can_approve = (
            bool(saved_email.strip())
            and not has_unsaved_email_changes
            and not bool(record["email_approved"])
        )

        approve_button = st.button(
            "✅ Approve Customer Email",
            type="primary",
            disabled=not can_approve,
            key=f"approve_email_{selected_id}"
        )

        if approve_button:

            updated = db.approve_customer_email(
                selected_id
            )

            if updated:

                st.success(
                    "Customer email approved successfully."
                )

                refresh_records()
                st.rerun()

        if (
            record["email_approved"]
            and not has_unsaved_email_changes
        ):

            st.download_button(
                "📥 Download Approved Email",
                data=saved_email.encode("utf-8"),
                file_name=(
                    f"agrichain_email_{selected_id}.txt"
                ),
                mime="text/plain",
                key=f"download_email_{selected_id}"
            )

            st.caption(
                "Approved for manual use. "
                "The email has not been sent."
            )

else:

    st.info(
        "Register a complaint to begin "
        "resolution management."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AgriChain AI | Resolution Management "
    "| SQLite Database | Capstone Project"
)
