import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned datasets
train_df = pd.read_csv(PROCESSED_DIR / "train_cleaned.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test_cleaned.csv")

# Encode complaint categories
encoder = LabelEncoder()

train_df["category_encoded"] = encoder.fit_transform(
    train_df["category"]
)

test_df["category_encoded"] = encoder.transform(
    test_df["category"]
)

# Save encoded datasets
train_df.to_csv(
    PROCESSED_DIR / "train_encoded.csv",
    index=False
)

test_df.to_csv(
    PROCESSED_DIR / "test_encoded.csv",
    index=False
)

# Save encoder for future predictions
joblib.dump(
    encoder,
    MODELS_DIR / "category_encoder.joblib"
)

print("\nAGRICHAIN - CATEGORY ENCODING")

print("\nCATEGORY MAPPING:")
for index, category in enumerate(encoder.classes_):
    print(f"{index}: {category}")

print("\nTRAINING DATA PREVIEW:")
print(train_df[["category", "category_encoded"]].head())

print("\nEncoding completed successfully!")