import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - Task 2
# Generate MiniLM Sentence Embeddings
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load datasets
train_df = pd.read_csv(PROCESSED_DIR / "train_balanced.csv")
val_df = pd.read_csv(PROCESSED_DIR / "validation_split.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test_encoded.csv")

print("\nAGRiCHAIN - GENERATING EMBEDDINGS")

# Load pretrained MiniLM model
print("\nLoading MiniLM model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert complaint text into embeddings
print("\nGenerating training embeddings...")

X_train = model.encode(
    train_df["complaint_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nGenerating validation embeddings...")

X_val = model.encode(
    val_df["complaint_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nGenerating test embeddings...")

X_test = model.encode(
    test_df["complaint_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# Labels
y_train = train_df["category_encoded"].to_numpy()
y_val = val_df["category_encoded"].to_numpy()
y_test = test_df["category_encoded"].to_numpy()

# Save embeddings and labels
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "X_test.npy", X_test)

np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "y_val.npy", y_val)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

# Display results
print("\nEMBEDDING SHAPES")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nLABEL SHAPES")

print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

print("\nEmbeddings saved successfully!")