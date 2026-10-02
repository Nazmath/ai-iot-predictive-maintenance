"""
Streamlit dashboard for AI + IoT Predictive Maintenance.

Run:
    streamlit run app.py
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_FILE = Path("models/machine_health_model.joblib")

FEATURES = ["temperature", "vibration", "current", "rpm"]


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"]


def generate_reading(fault_chance=0.18):
    fault = np.random.random() < fault_chance

    if fault:
        return {
            "temperature": float(np.clip(np.random.normal(93, 7), 80, 115)),
            "vibration": float(np.clip(np.random.normal(0.78, 0.16), 0.45, 1.30)),
            "current": float(np.clip(np.random.normal(6.9, 0.8), 5.2, 9.5)),
            "rpm": float(np.clip(np.random.normal(1130, 100), 800, 1300)),
        }

    return {
        "temperature": float(np.clip(np.random.normal(68, 6), 50, 82)),
        "vibration": float(np.clip(np.random.normal(0.23, 0.07), 0.05, 0.45)),
        "current": float(np.clip(np.random.normal(4.3, 0.55), 2.5, 5.8)),
        "rpm": float(np.clip(np.random.normal(1450, 45), 1300, 1550)),
    }


def build_chart(history, column, title, unit):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(history))),
            y=history[column],
            mode="lines+markers",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Reading",
        yaxis_title=unit,
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


st.set_page_config(
    page_title="AI + IoT Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 AI + IoT Predictive Maintenance")
st.caption("Laptop-only predictive maintenance prototype using simulated IoT sensors and Machine Learning.")

model = load_model()

if model is None:
    st.error(
        "Trained model not found. Run these commands first:\n\n"
        "python generate_dataset.py\n\n"
        "python train_model.py"
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    fault_chance = st.slider(
        "Simulated fault probability",
        min_value=0.0,
        max_value=0.50,
        value=0.18,
        step=0.01,
    )

if "history" not in st.session_state:
    st.session_state.history = []

reading = generate_reading(fault_chance)

row = pd.DataFrame([reading])[FEATURES]
prediction = int(model.predict(row)[0])
probability = float(model.predict_proba(row)[0][1])

st.session_state.history.append(
    {
        **reading,
        "prediction": prediction,
        "fault_probability": probability,
    }
)

st.session_state.history = st.session_state.history[-30:]
history_df = pd.DataFrame(st.session_state.history)

if prediction == 0:
    status_text = "🟢 NORMAL"
    status_message = "Machine operating within the learned normal range."
else:
    status_text = "🔴 FAULT DETECTED"
    status_message = "Abnormal sensor pattern detected. Inspect the machine."

st.subheader("Current Machine Status")
status_col, risk_col = st.columns(2)

with status_col:
    st.metric("Machine Health", status_text)

with risk_col:
    st.metric("Fault Probability", f"{probability:.1%}")

st.info(status_message)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Temperature", f"{reading['temperature']:.1f} °C")

with c2:
    st.metric("Vibration", f"{reading['vibration']:.2f} g")

with c3:
    st.metric("Current", f"{reading['current']:.2f} A")

with c4:
    st.metric("RPM", f"{reading['rpm']:.0f}")

st.divider()

st.subheader("Live Sensor Trends")

left, right = st.columns(2)

with left:
    st.plotly_chart(
        build_chart(history_df, "temperature", "Temperature Trend", "°C"),
        use_container_width=True,
    )
    st.plotly_chart(
        build_chart(history_df, "current", "Current Trend", "A"),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        build_chart(history_df, "vibration", "Vibration Trend", "g"),
        use_container_width=True,
    )
    st.plotly_chart(
        build_chart(history_df, "rpm", "RPM Trend", "RPM"),
        use_container_width=True,
    )

st.subheader("Recent Predictions")

display_df = history_df.copy()
display_df["status"] = display_df["prediction"].map({0: "NORMAL", 1: "FAULT"})
display_df["fault_probability"] = display_df["fault_probability"].map(
    lambda x: f"{x:.1%}"
)

st.dataframe(
    display_df[
        [
            "temperature",
            "vibration",
            "current",
            "rpm",
            "status",
            "fault_probability",
        ]
    ].tail(10).iloc[::-1],
    use_container_width=True,
)

st.caption("Educational simulation: sensor data is synthetic and not a substitute for validated industrial monitoring.")

if auto_refresh:
    time.sleep(2)
    st.rerun()
