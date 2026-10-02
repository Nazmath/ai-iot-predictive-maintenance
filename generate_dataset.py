"""
Generate a synthetic IoT predictive-maintenance dataset.

Run:
    python generate_dataset.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 5000

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "machine_data.csv"


def generate_normal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    temperature = rng.normal(68, 6, n).clip(50, 82)
    vibration = rng.normal(0.23, 0.07, n).clip(0.05, 0.45)
    current = rng.normal(4.3, 0.55, n).clip(2.5, 5.8)
    rpm = rng.normal(1450, 45, n).clip(1300, 1550)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "current": current,
            "rpm": rpm,
            "status": 0,
        }
    )


def generate_fault(n: int, rng: np.random.Generator) -> pd.DataFrame:
    temperature = rng.normal(93, 7, n).clip(80, 115)
    vibration = rng.normal(0.78, 0.16, n).clip(0.45, 1.30)
    current = rng.normal(6.9, 0.8, n).clip(5.2, 9.5)
    rpm = rng.normal(1130, 100, n).clip(800, 1300)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "current": current,
            "rpm": rpm,
            "status": 1,
        }
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    n_normal = N_SAMPLES // 2
    n_fault = N_SAMPLES - n_normal

    normal = generate_normal(n_normal, rng)
    fault = generate_fault(n_fault, rng)

    data = pd.concat([normal, fault], ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    data.insert(0, "machine_id", rng.integers(1, 11, len(data)))
    data.insert(1, "timestamp", pd.date_range("2026-01-01", periods=len(data), freq="min"))

    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset created: {OUTPUT_FILE}")
    print(f"Rows: {len(data)}")
    print("\nClass distribution:")
    print(data["status"].value_counts().rename(index={0: "Normal", 1: "Fault"}))


if __name__ == "__main__":
    main()
