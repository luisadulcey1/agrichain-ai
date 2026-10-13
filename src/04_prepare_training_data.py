import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load encoded training data
df = pd.read_csv(PROCESSED_DIR / "train_encoded.csv")

# Stratified train-validation split
train_data, val_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["category_encoded"]
)

# Save datasets
train_data.to_csv(
    PROCESSED_DIR / "train_split.csv",
    index=False
)

val_data.to_csv(
    PROCESSED_DIR / "validation_split.csv",
    index=False
)

print("\nAGRICHAIN - TRAIN / VALIDATION SPLIT")

print("\nTraining records:", len(train_data))
print("Validation records:", len(val_data))

print("\nTRAINING CATEGORY DISTRIBUTION")
print(train_data["category"].value_counts())

print("\nVALIDATION CATEGORY DISTRIBUTION")
print(val_data["category"].value_counts())

print("\nDatasets saved successfully!")