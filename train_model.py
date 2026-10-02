"""
Train the predictive-maintenance ML model.

Run:
    python train_model.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/machine_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

FEATURES = ["temperature", "vibration", "current", "rpm"]
TARGET = "status"


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run 'python generate_dataset.py' first."
        )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fault"],
    )
    matrix = confusion_matrix(y_test, predictions)

    model_path = MODEL_DIR / "machine_health_model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        model_path,
    )

    report_path = REPORT_DIR / "model_metrics.txt"
    report_path.write_text(
        f"AI + IoT Predictive Maintenance\n\n"
        f"Model: Random Forest Classifier\n"
        f"Test samples: {len(y_test)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{matrix}\n",
        encoding="utf-8",
    )

    print("=" * 55)
    print("MODEL TRAINING COMPLETE")
    print("=" * 55)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
