"""
CG4002 B02 - IMU Data Recorder via Zenoh
==========================================
Records IMU data from 3 ESP32 FireBeetles via Zenoh subscriber.
Logs labeled exercise data for training the AI model.

Workflow:
  1. Run this script
  2. Press a number key to select exercise (0-5)
  3. Perform the exercise — data is logged with that label
  4. Press another number to switch exercise, or 'q' to quit
  5. Output: a timestamped CSV log file in data/recordings/

Prerequisites:
  pip install eclipse-zenoh

Usage:
  python software/record_data.py

  Then on your keyboard:
    0 = high_knees
    1 = pushup
    2 = situp
    3 = lunge
    4 = squat
    5 = overhead_hold
    6 = unknown (unrecognised / wrong movement)
    p = pause (stop labeling, data not recorded)
    q = quit and save
"""

import json
import time
import os
import csv
import threading
from datetime import datetime

try:
    import zenoh
    ZENOH_AVAILABLE = True
except ImportError:
    ZENOH_AVAILABLE = False
    print("WARNING: zenoh not installed. Install with: pip install eclipse-zenoh")
    print("         Running in DEMO MODE (no actual Zenoh connection).\n")


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]

# Zenoh topic names — EDIT THESE to match your ESP32 topics
IMU_TOPICS = {
    "esp/imu1/esp1": "chest",     # IMU 1 → chest
    "esp/imu2/esp2": "wrist",     # IMU 2 → wrist
    "esp/imu3/esp3": "thigh",     # IMU 3 → thigh
}

# Fields to extract from each IMU JSON (drop tempC)
IMU_FIELDS = ["ax", "ay", "az", "gx", "gy", "gz"]

# Zenoh connection endpoint — EDIT to match your setup
ZENOH_ENDPOINT = "tcp/172.20.10.2:7447"


current_label = None       # None = paused, 0-6 = recording that class
data_buffer = []           # list of rows to write
lock = threading.Lock()
running = True


def on_sample(sample):
    """Called for every incoming Zenoh message from any IMU topic."""
    global current_label

    if current_label is None:
        return  # paused, don't record

    topic = str(sample.key_expr)
    if topic not in IMU_TOPICS:
        return

    sensor_name = IMU_TOPICS[topic]

    try:
        payload = json.loads(sample.payload.to_string())
    except (json.JSONDecodeError, AttributeError):
        try:
            payload = json.loads(str(sample.payload))
        except:
            return

    values = []
    for field in IMU_FIELDS:
        values.append(float(payload.get(field, 0.0)))

    timestamp = time.time()

    row = {
        "timestamp": timestamp,
        "sensor": sensor_name,
        "label": current_label,
        "label_name": CLASSES[current_label],
        "ax": values[0],
        "ay": values[1],
        "az": values[2],
        "gx": values[3],
        "gy": values[4],
        "gz": values[5],
    }

    with lock:
        data_buffer.append(row)


def keyboard_listener():
    """Listen for keyboard input to change labels."""
    global current_label, running

    while running:
        try:
            user_input = input().strip().lower()
        except EOFError:
            break

        if user_input == 'q':
            running = False
            print("\n  [QUIT] Stopping recording...")
            break
        elif user_input == 'p':
            current_label = None
            print("  [PAUSED] Not recording. Press 0-5 to start.")
        elif user_input in [str(i) for i in range(len(CLASSES))]:
            current_label = int(user_input)
            print(f"  [RECORDING] Label: {current_label} = {CLASSES[current_label]}")
        else:
            print(f"  Unknown input '{user_input}'. Use 0-5, p (pause), q (quit).")


def save_recording(data_buffer, output_path):
    """Save all recorded data to CSV."""
    if not data_buffer:
        print("  No data recorded!")
        return

    fieldnames = ["timestamp", "sensor", "label", "label_name",
                  "ax", "ay", "az", "gx", "gy", "gz"]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_buffer)

    from collections import Counter
    label_counts = Counter(row["label_name"] for row in data_buffer)

    print(f"\n  Recording saved: {output_path}")
    print(f"  Total samples: {len(data_buffer)}")
    print(f"  Per-class sample counts (individual IMU readings):")
    for cls in CLASSES:
        count = label_counts.get(cls, 0)
        print(f"    {cls}: {count}")


def main():
    print("=" * 65)
    print("  CG4002 B02 — IMU Data Recorder via Zenoh")
    print("=" * 65)
    print()
    print("  Exercise labels:")
    for i, cls in enumerate(CLASSES):
        print(f"    {i} = {cls}")
    print()
    print("  Controls:")
    print("    0-5  → Start recording that exercise")
    print("    p    → Pause recording")
    print("    q    → Quit and save")
    print()
    print(f"  IMU Topics:")
    for topic, sensor in IMU_TOPICS.items():
        print(f"    {topic} → {sensor}")
    print()

    os.makedirs("data/recordings", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/recordings/recording_{timestamp_str}.csv"

    if not ZENOH_AVAILABLE:
        print("  ERROR: zenoh module not available.")
        print("  Install with: pip install eclipse-zenoh")
        print("  Or record data manually and use convert_log_to_npy.py")
        return

    print(f"  Connecting to Zenoh at {ZENOH_ENDPOINT}...")
    try:
        config = zenoh.Config()
        config.insert_json5("connect/endpoints", json.dumps([ZENOH_ENDPOINT]))
        session = zenoh.open(config)
        print("  ✓ Zenoh connected!")
    except Exception as e:
        print(f"  ERROR connecting to Zenoh: {e}")
        print("  Make sure your Zenoh router is running and the endpoint is correct.")
        return

    subscribers = []
    for topic in IMU_TOPICS:
        sub = session.declare_subscriber(topic, on_sample)
        subscribers.append(sub)
        print(f"  ✓ Subscribed to: {topic}")

    print()
    print("  Ready! Press 0-5 to start recording, q to quit.")
    print("  " + "-" * 50)

    try:
        keyboard_listener()
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED]")

    for sub in subscribers:
        sub.undeclare()
    session.close()

    with lock:
        save_recording(data_buffer, output_path)

    print()
    print(f"  Next step: python software/convert_log_to_npy.py {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
