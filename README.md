# AI + IoT Predictive Maintenance

A laptop-only **AI + IoT simulation project** that generates virtual industrial sensor data, trains a machine-learning model, predicts machine health in real time, and displays the results in a Streamlit dashboard.

> **Important:** This version uses simulated sensors. No physical IoT hardware is required. The architecture can later be connected to ESP32/Raspberry Pi and real sensors.

## Project idea

The system simulates four machine parameters:

- Temperature (°C)
- Vibration (g)
- Current (A)
- RPM

The generated data contains normal and faulty operating conditions. A Random Forest classifier learns the relationship between these sensor values and machine health.

### Flow

```text
Virtual IoT Sensors
        |
        v
Synthetic Sensor Data
        |
        v
Data Preprocessing
        |
        v
Random Forest ML Model
        |
        v
Real-time Prediction
        |
        v
Streamlit Dashboard
        |
        +----> Normal / Fault
        |
        +----> Alert
```

## Features

- Virtual IoT sensor simulation
- Synthetic normal/fault dataset generation
- Data preprocessing and validation
- Random Forest classification
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Saved trained model
- Real-time sensor simulation
- Streamlit dashboard
- Machine health status
- Failure-risk probability
- Interactive sensor charts
- Optional MQTT-ready architecture for future hardware integration

## Tech stack

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Matplotlib

## Folder structure

```text
ai-iot-predictive-maintenance/
│
├── app.py
├── generate_dataset.py
├── sensor_simulator.py
├── train_model.py
├── predict.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
│
├── data/
│   └── machine_data.csv
│
├── models/
│   └── machine_health_model.joblib
│
└── reports/
    └── model_metrics.txt
```

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-iot-predictive-maintenance.git
cd ai-iot-predictive-maintenance
```

Or download the ZIP and open the project folder.

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Generate the dataset

```bash
python generate_dataset.py
```

This creates:

```text
data/machine_data.csv
```

## 5. Train the AI model

```bash
python train_model.py
```

This creates:

```text
models/machine_health_model.joblib
reports/model_metrics.txt
```

The training script also prints classification metrics and a confusion matrix.

## 6. Test a single prediction

```bash
python predict.py
```

You will see an example prediction such as:

```text
Sensor readings:
Temperature: 68.5 °C
Vibration: 0.24 g
Current: 4.35 A
RPM: 1452

Prediction: NORMAL
Fault probability: 2.4%
```

## 7. Start the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Dashboard

The dashboard shows:

- Current temperature
- Current vibration
- Current
- RPM
- AI prediction
- Fault probability
- Live sensor charts
- Machine status
- Recent readings

## How the AI works

The model receives:

```text
Temperature
Vibration
Current
RPM
```

and predicts:

```text
0 = Normal
1 = Fault
```

A Random Forest classifier is used because it is simple, interpretable, and works well for this type of tabular demonstration dataset.

## Example

Normal:

```text
Temperature = 67 °C
Vibration   = 0.21 g
Current     = 4.3 A
RPM         = 1460
             |
             v
          AI Model
             |
             v
        NORMAL
```

Fault:

```text
Temperature = 94 °C
Vibration   = 0.83 g
Current     = 7.2 A
RPM         = 1090
             |
             v
          AI Model
             |
             v
          FAULT
```

## Important project limitation

This repository is an **educational simulation**. The sensor values are synthetically generated and therefore model performance on this dataset does not represent performance on a real industrial machine.

For a real deployment, collect properly labeled sensor data from the target machines and validate the model under real operating conditions.

## Future scope

1. Replace virtual sensors with ESP32/Raspberry Pi.
2. Connect temperature, vibration and current sensors.
3. Use MQTT for real IoT communication.
4. Store real-time data in a database/cloud platform.
5. Add anomaly detection and remaining useful life (RUL) estimation.
6. Experiment with LSTM/deep-learning models for time-series data.
7. Add email/mobile notifications.
8. Deploy the dashboard online.

## Suggested hardware for the future version

- ESP32
- Temperature sensor such as DHT22/DS18B20
- Accelerometer/vibration sensor
- Current sensor
- RPM/encoder sensor
- Motor test setup
- Power supply

The current repository does **not** require any of these.

## Author

**Nazmath Pasha**

AI + IoT Predictive Maintenance — Laptop-only prototype.

## License

MIT License. See `LICENSE`.
