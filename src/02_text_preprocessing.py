import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(RAW_DIR / "complaints_train.csv")
test_df = pd.read_csv(RAW_DIR / "complaints_test.csv")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)

# Apply preprocessing
train_df["clean_text"] = train_df["complaint_text"].apply(clean_text)
test_df["clean_text"] = test_df["complaint_text"].apply(clean_text)

# Save processed datasets
train_df.to_csv(PROCESSED_DIR / "train_cleaned.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test_cleaned.csv", index=False)

# Preview results
print("\nAGRICHAIN - TEXT PREPROCESSING")

print("\nORIGINAL TEXT:")
print(train_df["complaint_text"].iloc[0])

print("\nCLEANED TEXT:")
print(train_df["clean_text"].iloc[0])

print("\nTRAINING RECORDS:", len(train_df))
print("TEST RECORDS:", len(test_df))

print("\nProcessed datasets saved successfully!")