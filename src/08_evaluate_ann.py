import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# AgriChain AI - ANN Evaluation

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Load test data
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

# Load trained model and category encoder
model = tf.keras.models.load_model(
    MODELS_DIR / "complaint_ann.keras"
)

encoder = joblib.load(
    MODELS_DIR / "category_encoder.joblib"
)

# Predict complaint categories
probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAGRiCHAIN - FINAL ANN EVALUATION")

print(f"\nTEST ACCURACY: {accuracy * 100:.2f}%")

# Precision, Recall and F1-score
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_))
)

fig, ax = plt.subplots(figsize=(11, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("AgriChain - ANN Confusion Matrix")
plt.tight_layout()

output_path = (
    PROCESSED_DIR / "ann_confusion_matrix.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("\nConfusion matrix saved successfully!")