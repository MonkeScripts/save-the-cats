"""
CG4002 B02 - Convert Real Training Data (InfluxDB CSV) to .npy
===============================================================
Parses InfluxDB annotated CSV exports from the Real_Training_Data folder,
extracts IMU accelerometer readings from all 3 sensors (arm, chest, thigh),
aligns them by timestamp, and produces windowed .npy arrays ready for training.

Data characteristics:
  - 3 sensors (arm, chest, thigh) with 4 features each: ax, ay, az, avm
  - 12 total features per timestep
  - ~8Hz effective sample rate
  - Each file = 30s recording of one exercise by one person

Usage:
  python convert_real_data.py
"""

import csv
import json
import numpy as np
import os
import sys
from collections import defaultdict


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES = len(CLASSES)

SENSOR_ORDER = ["arm", "chest", "thigh"]
FEATURES_PER_SENSOR = ["ax", "ay", "az", "avm"]
NUM_FEATURES = len(SENSOR_ORDER) * len(FEATURES_PER_SENSOR)  # 12

SAMPLE_RATE = 8             # ~8Hz effective rate from ESP
WINDOW_SIZE = 20            # ~2.5s at 8Hz
STRIDE = 10                 # 50% overlap

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATA_DIR = "data/Real_Training_Data"


FILENAME_TO_CLASS = {
    # First person
    "high-knee, 30s.csv": "high_knees",
    "push-up, 30s.csv": "pushup",
    "sit-up, 30s.csv": "situp",
    "lunges, 30s.csv": "lunge",
    "squats, 30s.csv": "squat",
    "overhead_arm, 30s.csv": "overhead_hold",
    "control-stationary, 30s.csv": "unknown",
    # Second person
    "v2high-knee, 30s.csv": "high_knees",
    "v2push-up, 30s.csv": "pushup",
    "v2sit-up, 30s.csv": "situp",
    "v2lunges, 30s.csv": "lunge",
    "v2squat, 30s.csv": "squat",
    "v2overhead_arm, 30s.csv": "overhead_hold",
    # Third person
    "v3high-knee.csv": "high_knees",
    "v3pushup.csv": "pushup",
    "v3situp.csv": "situp",
    "v3lunges.csv": "lunge",
    "v3squats.csv": "squat",
    "v3overhead_hold.csv": "overhead_hold",
    "v3control.csv": "unknown",
    # Fourth person
    "v4high-knee.csv": "high_knees",
    "v4pushup.csv": "pushup",
    "v4sit-up.csv": "situp",
    "v4lunges.csv": "lunge",
    "v4squats.csv": "squat",
    "v4overhead-hold.csv": "overhead_hold",
    "v4control.csv": "unknown",
    # Fifth person
    "v5highknee.csv": "high_knees",
    "v5pushup.csv": "pushup",
    "v5situp.csv": "situp",
    "v5lunges.csv": "lunge",
    "v5squat.csv": "squat",
    "v5armhold.csv": "overhead_hold",
    "v5control.csv": "unknown",
}


def parse_influxdb_csv(filepath):
    """
    Parse an InfluxDB annotated CSV and extract IMU data from all 3 sensors.

    Returns dict: { "arm": [...], "chest": [...], "thigh": [...] }
    Each list contains dicts with keys: time_str, t_ms, ax, ay, az, avm
    """
    sensor_data = {s: [] for s in SENSOR_ORDER}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ",value,esp/" not in line:
            continue

        sensor = None
        for s in SENSOR_ORDER:
            if f",value,esp/{s}," in line:
                sensor = s
                break
        if sensor is None:
            continue

        json_start = line.find('"{')
        json_end = line.find('}"', json_start)
        if json_start < 0 or json_end < 0:
            continue

        raw_json = line[json_start + 1:json_end + 1]
        raw_json = raw_json.replace('""', '"')

        prefix = line[:json_start]
        prefix_parts = prefix.rstrip(',').split(',')
        time_str = prefix_parts[5] if len(prefix_parts) > 5 else ""

        try:
            payload = json.loads(raw_json)
            sensor_data[sensor].append({
                "time_str": time_str,
                "t_ms": int(payload.get("t_ms", 0)),
                "ax": float(payload.get("ax", 0)),
                "ay": float(payload.get("ay", 0)),
                "az": float(payload.get("az", 0)),
                "avm": float(payload.get("avm", 0)),
            })
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    return sensor_data


def align_sensors(sensor_data):
    """
    Align 3 sensors by Zenoh receive time (_time) to create 12-feature rows.

    Since all 3 sensors publish at roughly the same rate (~8Hz), we pair them
    by their Zenoh arrival time. For each arm reading, find the nearest chest
    and thigh readings within a time tolerance.

    Returns list of dicts with 12 features (4 per sensor × 3 sensors).
    """
    arm = sensor_data["arm"]
    chest = sensor_data["chest"]
    thigh = sensor_data["thigh"]

    if not arm or not chest or not thigh:
        return []

    # Sort each by time_str (ISO timestamps sort lexicographically)
    arm.sort(key=lambda r: r["time_str"])
    chest.sort(key=lambda r: r["time_str"])
    thigh.sort(key=lambda r: r["time_str"])

    # Use arm as the reference timeline, find nearest chest/thigh for each
    aligned = []
    ci, ti = 0, 0  # chest and thigh indices

    for a in arm:
        a_time = a["time_str"]

        # Advance chest index to nearest
        while ci < len(chest) - 1 and chest[ci + 1]["time_str"] <= a_time:
            ci += 1

        # Advance thigh index to nearest
        while ti < len(thigh) - 1 and thigh[ti + 1]["time_str"] <= a_time:
            ti += 1

        c = chest[ci]
        t = thigh[ti]

        # Build 12-feature row: [arm_ax, arm_ay, arm_az, arm_avm,
        #                        chest_ax, chest_ay, chest_az, chest_avm,
        #                        thigh_ax, thigh_ay, thigh_az, thigh_avm]
        features = [
            a["ax"], a["ay"], a["az"], a["avm"],
            c["ax"], c["ay"], c["az"], c["avm"],
            t["ax"], t["ay"], t["az"], t["avm"],
        ]
        aligned.append(features)

    return aligned


def segment_windows(aligned_data, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Segment aligned data into fixed-size sliding windows.
    Returns array of shape (N, window_size, NUM_FEATURES).
    """
    if len(aligned_data) < window_size:
        return np.array([], dtype=np.float32).reshape(0, window_size, NUM_FEATURES)

    windows = []
    for start in range(0, len(aligned_data) - window_size + 1, stride):
        window = aligned_data[start:start + window_size]
        windows.append(np.array(window, dtype=np.float32))

    return np.array(windows, dtype=np.float32)


def main():
    print("=" * 65)
    print("  CG4002 B02 - Convert Real Training Data to .npy")
    print("=" * 65)

    all_files = []
    for person_dir in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_dir)
        if not os.path.isdir(person_path) or person_dir.startswith("."):
            continue
        for fname in sorted(os.listdir(person_path)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(person_path, fname)
            all_files.append((person_dir, fname, fpath))

    print(f"\n  Found {len(all_files)} CSV files across {len(set(f[0] for f in all_files))} people")

    all_X = []
    all_y = []
    all_person = []

    class_counts = defaultdict(int)

    for person_dir, fname, fpath in all_files:
        class_name = FILENAME_TO_CLASS.get(fname)
        if class_name is None:
            print(f"  WARNING: No class mapping for {fname}, skipping")
            continue

        class_idx = CLASSES.index(class_name)

        sensor_data = parse_influxdb_csv(fpath)

        sensor_counts = {s: len(sensor_data[s]) for s in SENSOR_ORDER}
        total_readings = sum(sensor_counts.values())
        if total_readings == 0:
            print(f"  WARNING: No data in {fpath}, skipping")
            continue

        aligned = align_sensors(sensor_data)
        if not aligned:
            print(f"  WARNING: Could not align sensors in {fname}")
            continue

        windows = segment_windows(aligned)
        if len(windows) == 0:
            print(f"  WARNING: Not enough aligned samples for a window in {fname} "
                  f"({len(aligned)} aligned, need {WINDOW_SIZE})")
            continue

        labels = np.full(len(windows), class_idx, dtype=np.int64)

        all_X.append(windows)
        all_y.append(labels)
        all_person.extend([person_dir] * len(windows))

        class_counts[class_name] += len(windows)
        print(f"  {person_dir:15s} / {fname:30s} -> {class_name:15s}  "
              f"arm={sensor_counts['arm']:3d} chest={sensor_counts['chest']:3d} "
              f"thigh={sensor_counts['thigh']:3d} -> {len(aligned):3d} aligned -> "
              f"{len(windows):3d} windows")

    if not all_X:
        print("\n  ERROR: No data processed!")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    persons = np.array(all_person)

    print(f"\n  Total: {len(X)} windows, shape X={X.shape}")
    print(f"\n  Class distribution:")
    for cls_name in CLASSES:
        count = class_counts.get(cls_name, 0)
        print(f"    {cls_name:15s}: {count:4d} windows")

    np.random.seed(42)
    perm = np.random.permutation(len(X))
    X, y, persons = X[perm], y[perm], persons[perm]

    n = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    np.save("data/train/X.npy", X_tr)
    np.save("data/train/y.npy", y_tr)
    np.save("data/val/X.npy", X_va)
    np.save("data/val/y.npy", y_va)
    np.save("data/test/X.npy", X_te)
    np.save("data/test/y.npy", y_te)

    print(f"\n  Saved:")
    print(f"    data/train/X.npy  {X_tr.shape}  y: {y_tr.shape}")
    print(f"    data/val/X.npy    {X_va.shape}  y: {y_va.shape}")
    print(f"    data/test/X.npy   {X_te.shape}  y: {y_te.shape}")

    meta = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "num_features": NUM_FEATURES,
        "features": [f"{s}_{f}" for s in SENSOR_ORDER for f in FEATURES_PER_SENSOR],
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "sample_rate": SAMPLE_RATE,
        "sensor_locations": SENSOR_ORDER,
        "train_total": int(len(X_tr)),
        "val_total": int(len(X_va)),
        "test_total": int(len(X_te)),
        "source": "real_imu_data",
        "source_files": [f"{p}/{f}" for p, f, _ in all_files],
    }
    with open("data/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Train class distribution:")
    for i, cls in enumerate(CLASSES):
        count = int(np.sum(y_tr == i))
        print(f"    {cls:15s}: {count:4d}")

    print(f"\n  Next steps:")
    print(f"    python train.py")
    print()
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
