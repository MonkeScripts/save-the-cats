"""
CG4002 B02 - Convert Recorded IMU Logs to Training Data (.npy)
===============================================================
Takes CSV files recorded by record_data.py (or manually logged Zenoh output)
and converts them into the (N, 128, 18) .npy format for model training.

How it works:
  1. Reads CSV with columns: timestamp, sensor, label, label_name, ax, ay, az, gx, gy, gz
  2. Groups rows by timestamp to align 3 IMUs into one 18-feature row
  3. Segments into sliding windows of 128 samples
  4. Splits into train (80%) and test (20%)
  5. Saves as data/train/X.npy, data/train/y.npy, etc.

The 18 features per timestep are ordered:
  [chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz,
   wrist_ax, wrist_ay, wrist_az, wrist_gx, wrist_gy, wrist_gz,
   thigh_ax, thigh_ay, thigh_az, thigh_gx, thigh_gy, thigh_gz]

Usage:
  python software/convert_log_to_npy.py data/recordings/recording_XXXXXX.csv

  Or to process ALL recordings in the folder:
  python software/convert_log_to_npy.py data/recordings/
"""

import csv
import numpy as np
import os
import sys
import json
import glob
from collections import defaultdict


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES = len(CLASSES)

SENSOR_ORDER = ["chest", "wrist", "thigh"]  # must match generate_data.py order
IMU_FIELDS = ["ax", "ay", "az", "gx", "gy", "gz"]
NUM_FEATURES = len(SENSOR_ORDER) * len(IMU_FIELDS)  # 18

WINDOW_SIZE = 128           # samples per window
STRIDE = 64                 # 50% overlap for more training data
TRAIN_RATIO = 0.8           # 80% train, 20% test

SAMPLE_RATE = 50            # expected Hz (for info only)

MAX_TIME_GAP = 0.1          # 100ms — roughly 2x the 50Hz period


def read_csv(filepath):
    """
    Read a recorded CSV file.
    Returns list of dicts with: timestamp, sensor, label, ax-gz values.
    """
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "sensor": row["sensor"],
                "label": int(row["label"]),
                "ax": float(row["ax"]),
                "ay": float(row["ay"]),
                "az": float(row["az"]),
                "gx": float(row["gx"]),
                "gy": float(row["gy"]),
                "gz": float(row["gz"]),
            })
    return rows


def read_zenoh_log(filepath):
    """
    Alternative: parse raw Zenoh subscriber log output.
    Handles lines like:
      >> [Subscriber] Received SampleKind.PUT ('esp/imu1/esp1':  '{"ax":0.07,"ay":0.35,...}')

    Requires a label file or manual annotation.
    Returns list of dicts.
    """
    rows = []
    # Topic-to-sensor mapping (edit if your topics differ)
    topic_map = {
        "esp/imu1/esp1": "chest",
        "esp/imu2/esp2": "wrist",
        "esp/imu3/esp3": "thigh",
    }

    with open(filepath, 'r') as f:
        for line in f:
            if "SampleKind.PUT" not in line:
                continue

            try:
                # Extract topic
                topic_start = line.index("('") + 2
                topic_end = line.index("'", topic_start)
                topic = line[topic_start:topic_end]

                if topic not in topic_map:
                    continue

                # Extract JSON payload
                json_start = line.index("{", topic_end)
                json_end = line.index("}", json_start) + 1
                payload = json.loads(line[json_start:json_end])

                rows.append({
                    "timestamp": len(rows) / SAMPLE_RATE,  # approximate timestamp
                    "sensor": topic_map[topic],
                    "label": -1,  # must be set manually or via label file
                    "ax": float(payload.get("ax", 0)),
                    "ay": float(payload.get("ay", 0)),
                    "az": float(payload.get("az", 0)),
                    "gx": float(payload.get("gx", 0)),
                    "gy": float(payload.get("gy", 0)),
                    "gz": float(payload.get("gz", 0)),
                })
            except (ValueError, json.JSONDecodeError, KeyError):
                continue

    return rows


def align_sensors(rows):
    """
    Group IMU readings by timestamp to create aligned 18-feature rows.

    Since the 3 ESPs send data independently, we group readings that
    arrive within a small time window and combine them into one row.

    Returns list of dicts: {timestamp, label, features[18]}
    """
    rows.sort(key=lambda r: r["timestamp"])

    BIN_SIZE = 1.0 / SAMPLE_RATE  # 0.02s = 20ms

    aligned = []
    i = 0
    while i < len(rows):
        # Collect all readings within this time bin
        bin_start = rows[i]["timestamp"]
        bin_data = {}   # sensor_name → {ax, ay, ...}
        bin_label = rows[i]["label"]

        while i < len(rows) and (rows[i]["timestamp"] - bin_start) < BIN_SIZE:
            r = rows[i]
            sensor = r["sensor"]
            if sensor in SENSOR_ORDER:
                bin_data[sensor] = [r[f] for f in IMU_FIELDS]
                bin_label = r["label"]
            i += 1

        if len(bin_data) == len(SENSOR_ORDER):
            features = []
            for sensor in SENSOR_ORDER:
                features.extend(bin_data[sensor])

            aligned.append({
                "timestamp": bin_start,
                "label": bin_label,
                "features": features,  # 18 values
            })

    return aligned


def segment_windows(aligned_data, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Segment aligned data into fixed-size sliding windows.
    Only creates windows where all samples have the SAME label.
    Respects time gaps (starts new segment if gap is too large).
    """
    windows_X = []
    windows_y = []

    segments = []
    current_segment = [aligned_data[0]]

    for i in range(1, len(aligned_data)):
        time_gap = aligned_data[i]["timestamp"] - aligned_data[i-1]["timestamp"]
        label_changed = aligned_data[i]["label"] != aligned_data[i-1]["label"]

        if time_gap > MAX_TIME_GAP or label_changed:
            segments.append(current_segment)
            current_segment = []

        current_segment.append(aligned_data[i])

    if current_segment:
        segments.append(current_segment)

    for segment in segments:
        if len(segment) < window_size:
            continue  # too short

        label = segment[0]["label"]
        if label < 0:
            continue  # unlabeled

        for start in range(0, len(segment) - window_size + 1, stride):
            window = segment[start:start + window_size]

            # Verify all same label
            if all(w["label"] == label for w in window):
                features = np.array([w["features"] for w in window], dtype=np.float32)
                windows_X.append(features)
                windows_y.append(label)

    return np.array(windows_X, dtype=np.float32), np.array(windows_y, dtype=np.int64)


def split_data(X, y, train_ratio=TRAIN_RATIO):
    """Shuffle and split into train/test."""
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    print("=" * 65)
    print("  CG4002 B02 — Convert IMU Logs to Training Data")
    print("=" * 65)

    if len(sys.argv) < 2:
        # Default: process all CSVs in recordings folder
        input_path = "data/recordings/"
        print(f"\n  No input specified. Looking in {input_path}")
    else:
        input_path = sys.argv[1]

    if os.path.isdir(input_path):
        csv_files = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        log_files = sorted(glob.glob(os.path.join(input_path, "*.log")))
        log_files += sorted(glob.glob(os.path.join(input_path, "*.txt")))
        all_files = csv_files + log_files
    else:
        all_files = [input_path]

    if not all_files:
        print(f"\n  ERROR: No CSV/log files found in {input_path}")
        print("  Record data first with: python software/record_data.py")
        return

    print(f"\n  Found {len(all_files)} file(s):")
    for f in all_files:
        print(f"    {f}")

    all_rows = []
    for filepath in all_files:
        print(f"\n  Reading {filepath}...")
        if filepath.endswith(".csv"):
            rows = read_csv(filepath)
        else:
            rows = read_zenoh_log(filepath)
        print(f"    {len(rows)} raw IMU readings")
        all_rows.extend(rows)

    if not all_rows:
        print("\n  ERROR: No data found in files!")
        return

    print(f"\n  Total raw readings: {len(all_rows)}")

    print("\n  Aligning 3 IMU sensors by timestamp...")
    aligned = align_sensors(all_rows)
    print(f"    Aligned rows (all 3 sensors present): {len(aligned)}")

    if not aligned:
        print("\n  ERROR: No aligned data! Check that all 3 sensors are present.")
        print("  Sensor names expected:", SENSOR_ORDER)
        return

    print(f"\n  Segmenting into {WINDOW_SIZE}-sample windows (stride={STRIDE})...")
    X, y = segment_windows(aligned, WINDOW_SIZE, STRIDE)
    print(f"    Windows created: {len(X)}")
    print(f"    Shape: X={X.shape}  y={y.shape}")

    if len(X) == 0:
        print("\n  ERROR: No windows created!")
        print(f"  Need at least {WINDOW_SIZE} consecutive aligned samples per exercise.")
        print(f"  You have {len(aligned)} aligned samples total.")
        return

    print(f"\n  Class distribution:")
    for i, cls in enumerate(CLASSES):
        count = np.sum(y == i)
        if count > 0:
            print(f"    {i} = {cls}: {count} windows")

    print(f"\n  Splitting {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f} train/test...")
    X_tr, y_tr, X_te, y_te = split_data(X, y)

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test",  exist_ok=True)

    np.save("data/train/X.npy", X_tr)
    np.save("data/train/y.npy", y_tr)
    np.save("data/test/X.npy", X_te)
    np.save("data/test/y.npy", y_te)

    print(f"\n  Saved:")
    print(f"    data/train/X.npy  {X_tr.shape}")
    print(f"    data/train/y.npy  {y_tr.shape}")
    print(f"    data/test/X.npy   {X_te.shape}")
    print(f"    data/test/y.npy   {y_te.shape}")

    meta = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "num_features": NUM_FEATURES,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "sample_rate": SAMPLE_RATE,
        "sensor_locations": SENSOR_ORDER,
        "axes_per_sensor": IMU_FIELDS,
        "train_total": int(len(X_tr)),
        "test_total": int(len(X_te)),
        "source": "real_imu_data",
        "source_files": [os.path.basename(f) for f in all_files],
    }
    with open("data/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Next steps:")
    print(f"    1. python software/train.py        (retrain model)")
    print(f"    2. python software/evaluate.py      (evaluate)")
    print()
    print("  ✓ Conversion complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
