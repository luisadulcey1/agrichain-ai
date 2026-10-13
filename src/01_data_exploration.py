import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# AgriChain AI Complaint Resolution System
# Task 1: Data Preparation and Exploration
# ---------------------------------------------------------

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Load datasets
train_path = RAW_DATA_DIR / "complaints_train.csv"
test_path = RAW_DATA_DIR / "complaints_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display basic dataset information
print("=" * 60)
print("AGRiCHAIN - DATASET OVERVIEW")
print("=" * 60)

print("\nTRAINING DATASET")
print(f"Rows: {train_df.shape[0]}")
print(f"Columns: {train_df.shape[1]}")

print("\nTEST DATASET")
print(f"Rows: {test_df.shape[0]}")
print(f"Columns: {test_df.shape[1]}")

print("\nCOLUMN NAMES")
print(train_df.columns.tolist())

print("\nFIRST 5 TRAINING RECORDS")
print(train_df.head())

print("\nDATA TYPES")
print(train_df.dtypes)

print("\nMISSING VALUES")
print(train_df.isnull().sum())

print("\nDUPLICATED ROWS")
print(train_df.duplicated().sum())

print("\nCOMPLAINT CATEGORY DISTRIBUTION")
print(train_df["category"].value_counts())

print("\nNUMBER OF CATEGORIES")
print(train_df["category"].nunique())

# ---------------------------------------------------------
# Complaint Category Distribution Visualization
# ---------------------------------------------------------

import matplotlib.pyplot as plt

category_counts = train_df["category"].value_counts()

plt.figure(figsize=(12, 6))

category_counts.plot(kind="bar")

plt.title("AgriChain - Complaint Category Distribution")
plt.xlabel("Complaint Category")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

# Save visualization
output_path = BASE_DIR / "data" / "processed" / "category_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print(f"\nChart saved successfully: {output_path}")

# ---------------------------------------------------------
# Complaint Distribution by Region and Channel
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Complaints by Region
train_df["region"].value_counts().plot(
    kind="bar",
    ax=axes[0],
    color="steelblue"
)

axes[0].set_title("Complaints by Region")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Number of Complaints")
axes[0].tick_params(axis="x", rotation=30)

# Complaints by Channel
train_df["channel"].value_counts().plot(
    kind="bar",
    ax=axes[1],
    color="seagreen"
)

axes[1].set_title("Complaints by Channel")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Number of Complaints")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()

output_path = BASE_DIR / "data" / "processed" / "region_channel_distribution.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("\nCOMPLAINTS BY REGION")
print(train_df["region"].value_counts())

print("\nCOMPLAINTS BY CHANNEL")
print(train_df["channel"].value_counts())

print(f"\nChart saved successfully: {output_path}")

# ---------------------------------------------------------
# Monthly Complaint Trends
# ---------------------------------------------------------

# Convert timestamp column to datetime
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

# Group complaints by month
monthly_complaints = (
    train_df.set_index("timestamp")
    .resample("MS")
    .size()
)

# Create line chart
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_complaints.index,
    monthly_complaints.values,
    marker="o",
    linewidth=2
)

plt.title("AgriChain - Monthly Complaint Trends")
plt.xlabel("Month")
plt.ylabel("Number of Complaints")
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save chart
output_path = BASE_DIR / "data" / "processed" / "monthly_complaint_trends.png"
plt.savefig(output_path, dpi=300)

plt.show()

print("\nMONTHLY COMPLAINT TRENDS")
print(monthly_complaints)

print(f"\nChart saved successfully: {output_path}")