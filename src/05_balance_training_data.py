import pandas as pd
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load training split only
train_df = pd.read_csv(
    PROCESSED_DIR / "train_split.csv"
)

print("\nAGRICHAIN - CLASS BALANCING")

print("\nBEFORE BALANCING:")
print(train_df["category"].value_counts())

# Balance training rows using their original indices
sampler = RandomOverSampler(random_state=42)

row_indices = train_df.index.to_numpy().reshape(-1, 1)
labels = train_df["category_encoded"]

balanced_indices, balanced_labels = sampler.fit_resample(
    row_indices,
    labels
)

# Recover complete rows, including complaint text
balanced_df = train_df.iloc[
    balanced_indices.flatten()
].copy()

# Shuffle balanced dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save balanced training dataset
output_path = PROCESSED_DIR / "train_balanced.csv"
balanced_df.to_csv(output_path, index=False)

print("\nAFTER BALANCING:")
print(balanced_df["category"].value_counts())

print("\nTOTAL TRAINING RECORDS:", len(balanced_df))
print("\nBalanced dataset saved successfully!")