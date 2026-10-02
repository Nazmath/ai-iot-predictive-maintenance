"""
Predict machine health from one sensor reading.

Run:
    python predict.py
"""

from pathlib import Path
import joblib
import pandas as pd

MODEL_FILE = Path("models/machine_health_model.joblib")


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run generate_dataset.py and train_model.py first."
        )

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]

    sample = pd.DataFrame(
        [
            {
                "temperature": 68.5,
                "vibration": 0.24,
                "current": 4.35,
                "rpm": 1452,
            }
        ]
    )[features]

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    fault_probability = probabilities[1]

    print("Sensor readings:")
    print(sample.to_string(index=False))
    print()

    if prediction == 0:
        print("Prediction: NORMAL")
    else:
        print("Prediction: FAULT")

    print(f"Fault probability: {fault_probability:.2%}")


if __name__ == "__main__":
    main()
