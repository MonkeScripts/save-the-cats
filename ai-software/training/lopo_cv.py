"""
CG4002 B02 - LOPO Cross-Validation on Previous (v1) Data
==========================================================
Leave-One-Person-Out evaluation on data/Real_Training_Data/Previous.
  - 5 people, 7 classes
  - 3 sensors (arm, chest, thigh), 4 features each (avm, ax, ay, az) = 12 raw features
  - Original rate ~8 Hz, resampled to uniform 8 Hz via interpolation

Usage:
  python lopo_previous.py
"""

import json
import re
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from model import ExerciseCNN, CLASSES
from features import augment_data, NUM_FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREV_DATA_DIR = "data/Real_Training_Data/Previous"

SENSOR_ORDER      = ["arm", "chest", "thigh"]
FEATURES_PER_SENSOR = ["avm", "ax", "ay", "az"]

SAMPLE_RATE = 8
WINDOW_SIZE = 20
STRIDE      = 5

NUM_CLASSES = len(CLASSES)

# Filename -> class mapping for all 5 people in the Previous folder
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

EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 0.001
WEIGHT_DECAY = 1e-4
VAL_RATIO    = 0.15


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_previous_csv(filepath):
    """
    Parse an InfluxDB annotated CSV from the Previous folder.
    Extracts avm, ax, ay, az per sensor (no gyroscope).
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
    """Resample sensor readings to a uniform target rate via linear interpolation."""
    if len(readings) < 2:
        return readings

    readings = sorted(readings, key=lambda r: r["t_ms"])
    t_orig = np.array([r["t_ms"] for r in readings], dtype=np.float64)
    t_new  = np.arange(t_orig[0], t_orig[-1], 1000.0 / target_hz)

    result = []
    for field in FEATURES_PER_SENSOR:
        vals = np.array([r[field] for r in readings], dtype=np.float64)
        result.append(np.interp(t_new, t_orig, vals))

    return [
        {"t_ms": float(t_new[i]), **{f: float(result[j][i]) for j, f in enumerate(FEATURES_PER_SENSOR)}}
        for i in range(len(t_new))
    ]


def align_and_window(sensor_data):
    """Align sensors by t_ms and segment into (N, WINDOW_SIZE, 12) windows."""
    arm   = sorted(sensor_data["arm"],   key=lambda r: r["t_ms"])
    chest = sorted(sensor_data["chest"], key=lambda r: r["t_ms"])
    thigh = sorted(sensor_data["thigh"], key=lambda r: r["t_ms"])

    if not arm or not chest or not thigh:
        return np.array([], dtype=np.float32).reshape(0, WINDOW_SIZE, NUM_FEATURES)

    chest_t = np.array([r["t_ms"] for r in chest])
    thigh_t = np.array([r["t_ms"] for r in thigh])

    aligned = []
    for a in arm:
        t  = a["t_ms"]
        ci = int(np.argmin(np.abs(chest_t - t)))
        ti = int(np.argmin(np.abs(thigh_t - t)))
        c, th = chest[ci], thigh[ti]
        row = [a[f] for f in FEATURES_PER_SENSOR] + \
              [c[f] for f in FEATURES_PER_SENSOR] + \
              [th[f] for f in FEATURES_PER_SENSOR]
        aligned.append(row)

    aligned = np.array(aligned, dtype=np.float32)
    if len(aligned) < WINDOW_SIZE:
        return np.array([], dtype=np.float32).reshape(0, WINDOW_SIZE, NUM_FEATURES)

    windows = []
    for start in range(0, len(aligned) - WINDOW_SIZE + 1, STRIDE):
        windows.append(aligned[start:start + WINDOW_SIZE])

    return np.array(windows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_persons():
    persons = {}
    for person_dir in sorted(os.listdir(PREV_DATA_DIR)):
        person_path = os.path.join(PREV_DATA_DIR, person_dir)
        if not os.path.isdir(person_path):
            continue

        X_list, y_list = [], []
        for fname in sorted(os.listdir(person_path)):
            if not fname.endswith(".csv"):
                continue
            class_name = FILENAME_TO_CLASS.get(fname)
            if class_name is None:
                print(f"  WARNING: No mapping for {person_dir}/{fname}, skipping")
                continue
            class_idx = CLASSES.index(class_name)
            fpath = os.path.join(person_path, fname)

            sensor_data = parse_previous_csv(fpath)
            if any(len(sensor_data[s]) < 10 for s in SENSOR_ORDER):
                print(f"  SKIP {person_dir}/{fname}: insufficient data")
                continue

            for s in SENSOR_ORDER:
                sensor_data[s] = upsample_sensor(sensor_data[s])

            windows = align_and_window(sensor_data)
            if len(windows) == 0:
                print(f"  SKIP {person_dir}/{fname}: not enough samples")
                continue

            X_list.append(windows)
            y_list.append(np.full(len(windows), class_idx, dtype=np.int64))
            print(f"  {person_dir:15s} / {fname:35s} -> {class_name:14s}  {len(windows)} windows")

        if X_list:
            persons[person_dir] = {
                "X": np.concatenate(X_list, axis=0),
                "y": np.concatenate(y_list, axis=0),
            }
            print(f"  --> {person_dir}: {persons[person_dir]['X'].shape[0]} windows total\n")

    return persons


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def normalize_fit(X_tr):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def train_model(X_tr, y_tr, X_va, y_va, num_features):
    model     = ExerciseCNN(num_features=num_features, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds    = model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_acc


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    return logits.argmax(dim=1).numpy()


def compute_confusion_matrix(y_true, y_pred):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm):
    metrics = {}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[i] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[i].sum())}
    return metrics


def plot_confusion_matrix(cm, title, save_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    short = [c[:12] for c in CLASSES]
    sns.heatmap(cm,      annot=True, fmt='d',    cmap='Blues',
                xticklabels=short, yticklabels=short, ax=ax1)
    sns.heatmap(cm_norm, annot=True, fmt='.1%',  cmap='Blues',
                xticklabels=short, yticklabels=short, ax=ax2)
    for ax, t in zip([ax1, ax2], ['Counts', 'Normalized']):
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title} ({t})')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main LOPO loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  CG4002 B02 - LOPO on Previous (v1) Data")
    print("=" * 65)
    print(f"  Data:    {PREV_DATA_DIR}")
    print(f"  Classes: {CLASSES}")
    print()

    print("  Loading all person data...")
    persons = load_all_persons()
    person_list = sorted(persons.keys())
    n_persons   = len(person_list)
    print(f"\n  {n_persons} people: {person_list}\n")

    fold_results = []
    cm_aggregate = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

    for fold_idx, test_person in enumerate(person_list):
        print(f"\n{'='*65}")
        print(f"  Fold {fold_idx+1}/{n_persons} - held out: {test_person}")
        print(f"{'='*65}")

        X_train_raw = np.concatenate(
            [persons[p]["X"] for p in person_list if p != test_person], axis=0
        )
        y_train = np.concatenate(
            [persons[p]["y"] for p in person_list if p != test_person], axis=0
        )
        X_test_raw = persons[test_person]["X"]
        y_test     = persons[test_person]["y"]

        mean, std = normalize_fit(X_train_raw)
        X_train_norm = ((X_train_raw - mean) / std).astype(np.float32)
        X_test_norm  = ((X_test_raw  - mean) / std).astype(np.float32)

        np.random.seed(42)
        perm = np.random.permutation(len(X_train_norm))
        X_train_norm, y_train = X_train_norm[perm], y_train[perm]
        n_val = int(len(X_train_norm) * VAL_RATIO)
        X_va, y_va = X_train_norm[:n_val],  y_train[:n_val]
        X_tr, y_tr = X_train_norm[n_val:],  y_train[n_val:]

        print(f"  Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_test_norm)}")
        print(f"  Training ({EPOCHS} epochs)...")

        model, best_val_acc = train_model(X_tr, y_tr, X_va, y_va, num_features=X_tr.shape[-1])
        print(f"  Best val acc: {best_val_acc*100:.1f}%")

        y_pred = predict(model, X_test_norm)
        cm     = compute_confusion_matrix(y_test, y_pred)
        cm_aggregate += cm

        acc      = cm.diagonal().sum() / cm.sum()
        mets     = compute_metrics(cm)
        macro_f1 = np.mean([mets[i]["f1"] for i in range(NUM_CLASSES)])

        print(f"\n  Test accuracy: {acc*100:.1f}%  macro-F1: {macro_f1:.3f}")
        print(f"  {'Class':<18} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Supp':>5}")
        print("  " + "-" * 46)
        for i in range(NUM_CLASSES):
            m = mets[i]
            print(f"  {CLASSES[i]:<18} {m['precision']:>6.3f} {m['recall']:>7.3f} "
                  f"{m['f1']:>6.3f} {m['support']:>5d}")

        os.makedirs("models/lopo_previous", exist_ok=True)
        plot_confusion_matrix(
            cm,
            title=f"Fold {fold_idx+1} - held out: {test_person}",
            save_path=f"models/lopo_previous/confusion_fold{fold_idx+1}.png"
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "test_person": test_person,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_class": {CLASSES[i]: mets[i] for i in range(NUM_CLASSES)},
        })

    # Aggregate results
    print(f"\n{'='*65}")
    print(f"  LOPO SUMMARY - Previous (v1) Data  [{n_persons} folds]")
    print(f"{'='*65}")

    accs = [r["accuracy"]  for r in fold_results]
    f1s  = [r["macro_f1"]  for r in fold_results]

    print(f"  {'Held-out person':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print("  " + "-" * 44)
    for r in fold_results:
        print(f"  {r['test_person']:<20} {r['accuracy']*100:>9.1f}% {r['macro_f1']:>10.3f}")
    print("  " + "-" * 44)
    print(f"  {'MEAN +/- STD':<20} "
          f"{np.mean(accs)*100:>8.1f}% +/- {np.std(accs)*100:.1f}%  "
          f"{np.mean(f1s):>6.3f} +/- {np.std(f1s):.3f}")

    plot_confusion_matrix(
        cm_aggregate,
        title="LOPO Aggregate - Previous (v1) data",
        save_path="models/lopo_previous/confusion_aggregate.png"
    )

    agg_acc  = cm_aggregate.diagonal().sum() / cm_aggregate.sum()
    agg_mets = compute_metrics(cm_aggregate)
    agg_f1   = np.mean([agg_mets[i]["f1"] for i in range(NUM_CLASSES)])

    print(f"\n  Aggregate accuracy: {agg_acc*100:.1f}%  macro-F1: {agg_f1:.3f}")
    print(f"\n  Aggregate per-class:")
    print(f"  {'Class':<18} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Supp':>5}")
    print("  " + "-" * 46)
    for i in range(NUM_CLASSES):
        m = agg_mets[i]
        print(f"  {CLASSES[i]:<18} {m['precision']:>6.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5d}")

    with open("models/lopo_previous/lopo_results.json", "w") as f:
        json.dump({
            "folds": fold_results,
            "mean_accuracy":      float(np.mean(accs)),
            "std_accuracy":       float(np.std(accs)),
            "mean_macro_f1":      float(np.mean(f1s)),
            "std_macro_f1":       float(np.std(f1s)),
            "aggregate_accuracy": float(agg_acc),
            "aggregate_macro_f1": float(agg_f1),
        }, f, indent=2)

    print(f"\n  Results saved: models/lopo_previous/lopo_results.json")
    print(f"  Plots saved:   models/lopo_previous/")
    print("=" * 65)


if __name__ == "__main__":
    main()
