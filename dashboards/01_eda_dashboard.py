
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# AGRICHAIN AI - TASK 4
# DASHBOARD 1: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgriChain AI | EDA Dashboard",
    page_icon="🌱",
    layout="wide"
)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR / "data" / "processed" / "train_cleaned.csv"
)

TEST_PATH = (
    BASE_DIR / "data" / "processed" / "test_cleaned.csv"
)

# ---------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------

@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


try:
    train_df, test_df = load_data()

except FileNotFoundError:
    st.error(
        "Dataset files were not found. "
        "Please check the data/processed folder."
    )
    st.stop()

required_columns = ["category", "region", "priority"]

missing_columns = [
    col for col in required_columns
    if col not in train_df.columns
]

if missing_columns:
    st.error(
        f"Missing dataset columns: {missing_columns}"
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🌱 AgriChain AI")

st.sidebar.markdown(
    "AI-Powered Complaint Resolution System"
)

st.sidebar.divider()

st.sidebar.info(
    "Dashboard 1: Exploratory Data Analysis"
)

st.sidebar.markdown(
    """
    **Dataset Information**

    Explore customer complaints using the
    real AgriChain training dataset.
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "AgriChain AI | Capstone Project"
)

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.title("🌱 AgriChain AI")

st.subheader(
    "Customer Complaint Analytics Dashboard"
)

st.markdown(
    """
    Explore customer complaints, identify
    common issues, and analyze complaint
    patterns using interactive filters.
    """
)

st.divider()

# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------

st.header("📊 Project Overview")

total_training = len(train_df)
total_testing = len(test_df)
total_categories = train_df["category"].nunique()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Complaints",
        f"{total_training:,}"
    )

with col2:
    st.metric(
        "Testing Complaints",
        f"{total_testing:,}"
    )

with col3:
    st.metric(
        "Complaint Categories",
        total_categories
    )

st.success(
    "Real AgriChain datasets loaded successfully!"
)

st.divider()

# ---------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

st.header("📈 Exploratory Data Analysis")

st.subheader("📊 Complaints by Category")

category_counts = (
    train_df["category"]
    .value_counts()
    .sort_values(ascending=False)
)

st.bar_chart(
    category_counts,
    x_label="Complaint Category",
    y_label="Number of Complaints"
)

st.markdown(
    """
    This chart displays the distribution
    of complaints across categories in
    the complete training dataset.
    """
)

st.divider()

# ---------------------------------------------------------
# CATEGORY SUMMARY
# ---------------------------------------------------------

st.subheader("📋 Category Summary")

category_summary = (
    train_df["category"]
    .value_counts()
    .reset_index()
)

category_summary.columns = [
    "Complaint Category",
    "Total Complaints"
]

category_summary["Percentage"] = (
    category_summary["Total Complaints"]
    / total_training
    * 100
).round(2)

st.dataframe(
    category_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# INTERACTIVE FILTERS
# ---------------------------------------------------------

st.header("🔎 Interactive Complaint Explorer")

st.markdown(
    """
    Filter complaints by category, region,
    and priority to explore specific
    customer service patterns.
    """
)

category_options = ["All Categories"] + sorted(
    train_df["category"].dropna().unique().tolist()
)

region_options = ["All Regions"] + sorted(
    train_df["region"].dropna().unique().tolist()
)

priority_options = ["All Priorities"] + sorted(
    train_df["priority"].dropna().unique().tolist()
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_category = st.selectbox(
        "Complaint Category",
        category_options
    )

with filter_col2:
    selected_region = st.selectbox(
        "Region",
        region_options
    )

with filter_col3:
    selected_priority = st.selectbox(
        "Priority",
        priority_options
    )

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

filtered_df = train_df.copy()

if selected_category != "All Categories":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]

if selected_region != "All Regions":
    filtered_df = filtered_df[
        filtered_df["region"] == selected_region
    ]

if selected_priority != "All Priorities":
    filtered_df = filtered_df[
        filtered_df["priority"] == selected_priority
    ]

# ---------------------------------------------------------
# FILTERED RESULTS
# ---------------------------------------------------------

st.subheader("📌 Filtered Results")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.metric(
        "Matching Complaints",
        len(filtered_df)
    )

with result_col2:
    percentage = (
        len(filtered_df) / total_training * 100
        if total_training > 0 else 0
    )

    st.metric(
        "Percentage of Training Dataset",
        f"{percentage:.1f}%"
    )

if filtered_df.empty:
    st.warning(
        "No complaints match the selected filters."
    )

else:
    st.subheader("📊 Filtered Category Distribution")

    filtered_counts = (
        filtered_df["category"]
        .value_counts()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        filtered_counts,
        x_label="Complaint Category",
        y_label="Number of Complaints"
    )

    st.subheader("📋 Filtered Complaint Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

st.divider()


# ---------------------------------------------------------
# DOWNLOAD FILTERED COMPLAINTS
# ---------------------------------------------------------

st.subheader("📥 Export Complaint Data")

st.markdown(
    "Download the complaints matching your selected filters."
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Complaints (CSV)",
    data=csv_data,
    file_name="agrichain_filtered_complaints.csv",
    mime="text/csv",
    disabled=filtered_df.empty
)

st.divider()



# ---------------------------------------------------------
# MONTHLY COMPLAINT TRENDS
# ---------------------------------------------------------

st.header("📅 Monthly Complaint Trends")

st.markdown(
    "Explore how customer complaint volumes change over time."
)

monthly_df = filtered_df.copy()

monthly_df["timestamp"] = pd.to_datetime(
    monthly_df["timestamp"],
    errors="coerce"
)

monthly_df = monthly_df.dropna(subset=["timestamp"])

monthly_df["month"] = (
    monthly_df["timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_counts = (
    monthly_df.groupby("month")
    .size()
    .reset_index(name="complaints")
    .sort_values("month")
)

if monthly_counts.empty:
    st.info("No monthly data available for these filters.")
else:
    st.line_chart(
        monthly_counts,
        x="month",
        y="complaints",
        x_label="Month",
        y_label="Number of Complaints"
    )


# ---------------------------------------------------------
# DATASET QUALITY
# ---------------------------------------------------------

st.subheader("🧹 Dataset Quality Overview")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    st.metric(
        "Total Columns",
        len(train_df.columns)
    )

with quality_col2:
    st.metric(
        "Missing Values",
        int(train_df.isnull().sum().sum())
    )

with quality_col3:
    st.metric(
        "Duplicate Rows",
        int(train_df.duplicated().sum())
    )

st.divider()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.caption(
    "AgriChain AI Complaint Resolution System "
    "| EDA Dashboard | Capstone Project"
)
