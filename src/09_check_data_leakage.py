import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

train_df = pd.read_csv(DATA_DIR / "complaints_train.csv")
test_df = pd.read_csv(DATA_DIR / "complaints_test.csv")

# Normalize text for exact duplicate detection
train_texts = (
    train_df["complaint_text"]
    .astype(str)
    .str.lower()
    .str.strip()
)

test_texts = (
    test_df["complaint_text"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Check overlap between training and test
duplicates = test_texts.isin(set(train_texts))

print("\nAGRiCHAIN - DATA LEAKAGE CHECK")

print("\nTraining records:", len(train_df))
print("Test records:", len(test_df))

print("\nExact duplicate complaint texts:")
print(duplicates.sum())

print("\nDuplicate percentage:")
print(f"{duplicates.mean() * 100:.2f}%")

print("\nUnique training texts:", train_texts.nunique())
print("Unique test texts:", test_texts.nunique())

print("\nCheck completed successfully!")