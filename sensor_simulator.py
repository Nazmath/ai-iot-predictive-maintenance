"""
Virtual IoT sensor simulator.

This script does not require hardware. It generates realistic-looking
temperature, vibration, current and RPM values for a machine.

Run:
    python sensor_simulator.py

Press Ctrl+C to stop.
"""

import random
import time


def generate_reading(fault=False):
    if not fault:
        return {
            "temperature": random.gauss(68, 5),
            "vibration": max(0.05, random.gauss(0.23, 0.06)),
            "current": max(2.5, random.gauss(4.3, 0.5)),
            "rpm": max(1300, random.gauss(1450, 40)),
        }

    return {
        "temperature": random.gauss(93, 7),
        "vibration": max(0.45, random.gauss(0.78, 0.15)),
        "current": max(5.2, random.gauss(6.9, 0.8)),
        "rpm": max(800, random.gauss(1130, 90)),
    }


def main():
    print("Virtual IoT Sensor Simulator")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Mostly normal operation, with an occasional simulated fault.
            fault = random.random() < 0.15
            reading = generate_reading(fault)

            print(
                f"Temperature: {reading['temperature']:.2f} °C | "
                f"Vibration: {reading['vibration']:.2f} g | "
                f"Current: {reading['current']:.2f} A | "
                f"RPM: {reading['rpm']:.0f} | "
                f"Actual condition: {'FAULT' if fault else 'NORMAL'}"
            )

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
