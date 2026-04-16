"""
CG4002 B02 - Convert Previous Training Data (InfluxDB CSV) to .npy
===================================================================
Parses InfluxDB annotated CSV exports from Real_Training_Data/Previous,
extracts accelerometer readings from 3 sensors (arm, chest, thigh),
aligns them by timestamp, and produces windowed .npy arrays for training.

Data characteristics:
  - 3 sensors (arm, chest, thigh), 4 features each: avm, ax, ay, az
  - 12 total features per timestep
  - ~8 Hz effective sample rate (original recordings)
  - 5 people, 7 exercise classes
  - Each file = ~30s recording of one exercise by one person

Classes:
  0 = high_knees
  1 = lunge
  2 = squat
  3 = overhead_arm
  4 = push_up
  5 = sit_up
  6 = unknown

Usage:
  python convert_real_data.py
"""

import csv
import json
import re
import numpy as np
import os
from collections import defaultdict

from features import augment_data, NUM_FEATURES


CLASSES = ["high_knees", "lunge", "squat", "overhead_arm", "push_up", "sit_up", "unknown"]
NUM_CLASSES = len(CLASSES)

SENSOR_ORDER = ["arm", "chest", "thigh"]
FEATURES_PER_SENSOR = ["avm", "ax", "ay", "az"]

SAMPLE_RATE = 8             # ~8 Hz effective rate
WINDOW_SIZE = 20            # ~2.5s at 8 Hz
STRIDE      = 5             # 75% overlap

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

DATA_DIR = "data/Real_Training_Data/Previous"


# Filename-to-class mapping for all 5 people
FILENAME_TO_CLASS = {
    # first person
    "high-knee, 30s.csv":            "high_knees",
    "lunges, 30s.csv":               "lunge",
    "squats, 30s.csv":               "squat",
    "overhead_arm, 30s.csv":         "overhead_arm",
    "push-up, 30s.csv":              "push_up",
    "sit-up, 30s.csv":               "sit_up",
    "control-stationary, 30s.csv":   "unknown",
    # second person
    "v2high-knee, 30s.csv":          "high_knees",
    "v2lunges, 30s.csv":             "lunge",
    "v2squat, 30s.csv":              "squat",
    "v2overhead_arm, 30s.csv":       "overhead_arm",
    "v2push-up, 30s.csv":            "push_up",
    "v2sit-up, 30s.csv":             "sit_up",
    # third person
    "v3high-knee.csv":               "high_knees",
    "v3lunges.csv":                  "lunge",
    "v3squats.csv":                  "squat",
    "v3overhead_hold.csv":           "overhead_arm",
    "v3pushup.csv":                  "push_up",
    "v3situp.csv":                   "sit_up",
    "v3control.csv":                 "unknown",
    # fourth person
    "v4high-knee.csv":               "high_knees",
    "v4lunges.csv":                  "lunge",
    "v4squats.csv":                  "squat",
    "v4overhead-hold.csv":           "overhead_arm",
    "v4pushup.csv":                  "push_up",
    "v4sit-up.csv":                  "sit_up",
    "v4control.csv":                 "unknown",
    # fifth person
    "v5highknee.csv":                "high_knees",
    "v5lunges.csv":                  "lunge",
    "v5squat.csv":                   "squat",
    "v5armhold.csv":                 "overhead_arm",
    "v5pushup.csv":                  "push_up",
    "v5situp.csv":                   "sit_up",
    "v5control.csv":                 "unknown",
}


def parse_previous_csv(filepath):
    """
    Parse an InfluxDB annotated CSV from the Previous folder.

    Each data row contains a JSON payload with keys: t_ms, ax, ay, az, avm.
    The 'device' column identifies the sensor (esp/arm, esp/chest, esp/thigh).

    Returns dict: { "arm": [...], "chest": [...], "thigh": [...] }
    Each list contains dicts with keys: t_ms, avm, ax, ay, az
    """
    sensor_data = {s: [] for s in SENSOR_ORDER}

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        if '"ax"' not in line:
            continue

        sensor = None
        for s in SENSOR_ORDER:
            if f"esp/{s}" in line:
                sensor = s
                break
        if sensor is None:
            continue

        js = re.search(r'"(\{.*?\})"', line)
        if not js:
            continue

        raw = js.group(1).replace('""', '"')
        try:
            payload = json.loads(raw)
            sensor_data[sensor].append({
                "t_ms": int(payload["t_ms"]),
                "avm":  float(payload["avm"]),
                "ax":   float(payload["ax"]),
                "ay":   float(payload["ay"]),
                "az":   float(payload["az"]),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return sensor_data


def upsample_sensor(readings, target_hz=SAMPLE_RATE):
    """
    Resample sensor readings to a uniform target rate via linear interpolation.
    The original data is at approximately 8 Hz but with jitter.
    """
    if len(readings) < 2:
        return readings

    readings = sorted(readings, key=lambda r: r["t_ms"])
    t_orig = np.array([r["t_ms"] for r in readings], dtype=np.float64)
    t_start, t_end = t_orig[0], t_orig[-1]
    dt_ms = 1000.0 / target_hz
    t_new = np.arange(t_start, t_end, dt_ms)

    upsampled_fields = {}
    for field in FEATURES_PER_SENSOR:
        vals = np.array([r[field] for r in readings], dtype=np.float64)
        upsampled_fields[field] = np.interp(t_new, t_orig, vals)

    result = []
    for i, t in enumerate(t_new):
        entry = {"t_ms": float(t)}
        for field in FEATURES_PER_SENSOR:
            entry[field] = float(upsampled_fields[field][i])
        result.append(entry)

    return result


def align_sensors(sensor_data):
    """
    Align 3 sensors by timestamp and produce 12-feature rows.

    Uses arm as the reference timeline; for each arm reading, finds the
    nearest chest and thigh readings by t_ms.

    Returns list of 12-element feature vectors [arm_feats, chest_feats, thigh_feats].
    """
    arm   = sorted(sensor_data["arm"],   key=lambda r: r["t_ms"])
    chest = sorted(sensor_data["chest"], key=lambda r: r["t_ms"])
    thigh = sorted(sensor_data["thigh"], key=lambda r: r["t_ms"])

    if not arm or not chest or not thigh:
        return []

    chest_t = np.array([r["t_ms"] for r in chest])
    thigh_t = np.array([r["t_ms"] for r in thigh])

    aligned = []
    for a in arm:
        t = a["t_ms"]
        ci = int(np.argmin(np.abs(chest_t - t)))
        ti = int(np.argmin(np.abs(thigh_t - t)))
        c, th = chest[ci], thigh[ti]
        row = [a[f] for f in FEATURES_PER_SENSOR] + \
              [c[f] for f in FEATURES_PER_SENSOR] + \
              [th[f] for f in FEATURES_PER_SENSOR]
        aligned.append(row)

    return aligned


def segment_windows(aligned_data):
    """
    Segment aligned data into fixed-size sliding windows.

    Returns array of shape (N, WINDOW_SIZE, NUM_FEATURES).
    """
    if len(aligned_data) < WINDOW_SIZE:
        return np.array([], dtype=np.float32).reshape(0, WINDOW_SIZE, NUM_FEATURES)

    windows = []
    for start in range(0, len(aligned_data) - WINDOW_SIZE + 1, STRIDE):
        windows.append(np.array(aligned_data[start:start + WINDOW_SIZE], dtype=np.float32))

    return np.array(windows, dtype=np.float32)


def main():
    print("=" * 65)
    print("  CG4002 B02 - Convert Previous Training Data to .npy")
    print("=" * 65)

    all_files = []
    for person_dir in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_dir)
        if not os.path.isdir(person_path) or person_dir.startswith("."):
            continue
        for fname in sorted(os.listdir(person_path)):
            if not fname.endswith(".csv"):
                continue
            all_files.append((person_dir, fname, os.path.join(person_path, fname)))

    n_people = len(set(f[0] for f in all_files))
    print(f"\n  Found {len(all_files)} CSV files across {n_people} people")

    all_X, all_y = [], []
    class_counts = defaultdict(int)

    for person_dir, fname, fpath in all_files:
        class_name = FILENAME_TO_CLASS.get(fname)
        if class_name is None:
            print(f"  WARNING: No class mapping for {fname}, skipping")
            continue

        class_idx = CLASSES.index(class_name)
        sensor_data = parse_previous_csv(fpath)

        if any(len(sensor_data[s]) < 10 for s in SENSOR_ORDER):
            print(f"  SKIP {person_dir}/{fname}: insufficient sensor data")
            continue

        # Resample to uniform 8 Hz
        for s in SENSOR_ORDER:
            sensor_data[s] = upsample_sensor(sensor_data[s])

        aligned = align_sensors(sensor_data)
        if not aligned:
            print(f"  WARNING: Could not align sensors in {fname}")
            continue

        windows = segment_windows(aligned)
        if len(windows) == 0:
            print(f"  WARNING: Not enough samples in {fname} "
                  f"({len(aligned)} aligned, need {WINDOW_SIZE})")
            continue

        labels = np.full(len(windows), class_idx, dtype=np.int64)
        all_X.append(windows)
        all_y.append(labels)
        class_counts[class_name] += len(windows)
        print(f"  {person_dir:20s} / {fname:35s} -> {class_name:14s}  {len(windows)} windows")

    if not all_X:
        print("\n  ERROR: No data processed!")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\n  Total: {len(X)} windows, shape X={X.shape}")
    print(f"\n  Class distribution:")
    for cls_name in CLASSES:
        count = class_counts.get(cls_name, 0)
        print(f"    {cls_name:<18}: {count:4d} windows")

    # Random 70/15/15 split
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    X_tr, y_tr = X[:n_train],              y[:n_train]
    X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_te, y_te = X[n_train+n_val:],        y[n_train+n_val:]

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val",   exist_ok=True)
    os.makedirs("data/test",  exist_ok=True)

    np.save("data/train/X.npy", X_tr)
    np.save("data/train/y.npy", y_tr)
    np.save("data/val/X.npy",   X_va)
    np.save("data/val/y.npy",   y_va)
    np.save("data/test/X.npy",  X_te)
    np.save("data/test/y.npy",  y_te)

    print(f"\n  Saved:")
    print(f"    data/train/X.npy  {X_tr.shape}")
    print(f"    data/val/X.npy    {X_va.shape}")
    print(f"    data/test/X.npy   {X_te.shape}")
    print(f"\n  Next step: python train.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
