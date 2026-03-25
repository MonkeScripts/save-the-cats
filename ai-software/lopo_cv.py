"""
CG4002 B02 - Leave-One-Person-Out (LOPO) Cross-Validation
==========================================================
Trains 4 folds, each time holding out one person's data entirely.
Tests whether the model generalizes to unseen users, not just unseen windows.

Usage:
  python lopo_cv.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ExerciseCNN
from convert_real_data import (
    CLASSES, NUM_FEATURES, WINDOW_SIZE, STRIDE, SENSOR_ORDER,
    FILENAME_TO_CLASS, DATA_DIR,
    parse_influxdb_csv, align_sensors, segment_windows,
)


CONFIG = {
    "num_features": NUM_FEATURES,
    "num_classes": len(CLASSES),
    "window_size": WINDOW_SIZE,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epochs": 50,
}

PERSON_DIRS = ["first person", "second person", "third person", "fourth person", "fifth person"]

# Augmentation multiplier: how many augmented copies per original sample
AUG_COPIES = 4


def augment_scaling(X, scale_range=(0.8, 1.2)):
    """Random per-channel amplitude scaling."""
    scales = np.random.uniform(scale_range[0], scale_range[1], size=(1, X.shape[-1]))
    return X * scales


def augment_noise(X, noise_std=0.05):
    """Add Gaussian noise."""
    return X + np.random.normal(0, noise_std, X.shape)


def augment_time_warp(X, sigma=0.2):
    """
    Smooth time warping: generate a random cumulative warp path
    and resample the signal along it using linear interpolation.
    """
    T = X.shape[0]
    # Random warp: cumulative sum of 1 + small perturbations
    warp = np.cumsum(np.random.normal(1.0, sigma, T))
    # Normalize to [0, T-1]
    warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (T - 1)
    orig_indices = np.arange(T)
    warped = np.zeros_like(X)
    for ch in range(X.shape[-1]):
        warped[:, ch] = np.interp(orig_indices, warp, X[:, ch])
    return warped


def augment_rotation(X, num_sensors=3, features_per_sensor=4):
    """
    Apply a small random 3D rotation to each sensor's (ax, ay, az).
    avm is recomputed from the rotated axes.
    """
    X_aug = X.copy()
    for s in range(num_sensors):
        offset = s * features_per_sensor
        ax, ay, az = X[:, offset], X[:, offset+1], X[:, offset+2]

        # Small random rotation angles (radians)
        angles = np.random.uniform(-0.3, 0.3, 3)
        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])

        # Combined rotation matrix (Rz * Ry * Rx)
        R = np.array([
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy,   sx*cy,            cx*cy           ],
        ])

        accel = np.stack([ax, ay, az], axis=1)  # (T, 3)
        rotated = accel @ R.T  # (T, 3)

        X_aug[:, offset]   = rotated[:, 0]
        X_aug[:, offset+1] = rotated[:, 1]
        X_aug[:, offset+2] = rotated[:, 2]
        # Recompute avm
        X_aug[:, offset+3] = np.sqrt(rotated[:, 0]**2 + rotated[:, 1]**2 + rotated[:, 2]**2)

    return X_aug


def augment_time_shift(X, max_shift=3):
    """Circular time shift."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=0)


def augment_sample(X):
    """Apply a random combination of augmentations to one window."""
    X_aug = X.copy()

    # Always apply scaling + noise (mild, always helpful)
    X_aug = augment_scaling(X_aug)
    X_aug = augment_noise(X_aug)

    # Randomly apply stronger augmentations
    if np.random.random() < 0.5:
        X_aug = augment_time_warp(X_aug)
    if np.random.random() < 0.5:
        X_aug = augment_rotation(X_aug)
    if np.random.random() < 0.5:
        X_aug = augment_time_shift(X_aug)

    return X_aug.astype(np.float32)


def augment_dataset(X, y, num_copies=AUG_COPIES):
    """Create augmented copies of the entire training set."""
    X_aug_list = [X]  # keep originals
    y_aug_list = [y]

    for _ in range(num_copies):
        X_copy = np.array([augment_sample(x) for x in X])
        X_aug_list.append(X_copy)
        y_aug_list.append(y.copy())

    return np.concatenate(X_aug_list, axis=0), np.concatenate(y_aug_list, axis=0)


def load_all_by_person():
    """Load and segment all real data, grouped by person directory."""
    person_data = {}  # person_dir -> (X, y)

    for person_dir in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_dir)
        if not os.path.isdir(person_path) or person_dir.startswith("."):
            continue

        X_list, y_list = [], []

        for fname in sorted(os.listdir(person_path)):
            if not fname.endswith(".csv"):
                continue

            class_name = FILENAME_TO_CLASS.get(fname)
            if class_name is None:
                print(f"  WARNING: No class mapping for {fname}, skipping")
                continue

            class_idx = CLASSES.index(class_name)
            fpath = os.path.join(person_path, fname)

            sensor_data = parse_influxdb_csv(fpath)
            aligned = align_sensors(sensor_data)
            if not aligned:
                continue

            windows = segment_windows(aligned)
            if len(windows) == 0:
                continue

            X_list.append(windows)
            y_list.append(np.full(len(windows), class_idx, dtype=np.int64))

        if X_list:
            person_data[person_dir] = (
                np.concatenate(X_list, axis=0),
                np.concatenate(y_list, axis=0),
            )
            print(f"  {person_dir:15s}: {len(person_data[person_dir][0]):4d} windows")

    return person_data


def normalize_split(X_tr, X_te):
    """Z-score normalize using training stats only."""
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X_tr - mean) / std, (X_te - mean) / std


def train_fold(X_tr, y_tr, X_te, y_te, fold_name, epochs=CONFIG["epochs"]):
    """Train one LOPO fold and return test metrics."""
    X_tr, y_tr = augment_dataset(X_tr, y_tr)
    print(f"    After augmentation: {len(X_tr)} training windows ({AUG_COPIES}x copies + originals)")

    X_tr, X_te = normalize_split(X_tr, X_te)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr.astype(np.float32)),
        torch.from_numpy(y_tr),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_te.astype(np.float32)),
        torch.from_numpy(y_te),
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = ExerciseCNN(num_features=CONFIG["num_features"], num_classes=CONFIG["num_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
        tr_loss, tr_acc = total_loss / total, correct / total

        # Evaluate
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.numpy())
                all_labels.extend(y_batch.numpy())
        te_loss, te_acc = total_loss / total, correct / total

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_preds = np.array(all_preds)
            best_labels = np.array(all_labels)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:3d}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")

    return best_acc, best_preds, best_labels, history


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_lopo_results(fold_results, save_dir):
    """Plot per-fold training curves and confusion matrices."""
    os.makedirs(save_dir, exist_ok=True)
    num_folds = len(fold_results)
    num_classes = len(CLASSES)

    fig, axes = plt.subplots(2, num_folds, figsize=(5 * num_folds, 8))
    for i, (person, result) in enumerate(fold_results.items()):
        h = result["history"]
        epochs = range(1, len(h["train_loss"]) + 1)

        ax_loss = axes[0][i] if num_folds > 1 else axes[0]
        ax_loss.plot(epochs, h["train_loss"], 'b-', label='Train')
        ax_loss.plot(epochs, h["test_loss"], 'r-', label='Test')
        ax_loss.set_title(f'Loss (held out: {person})', fontsize=10)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        ax_acc = axes[1][i] if num_folds > 1 else axes[1]
        ax_acc.plot(epochs, h["train_acc"], 'b-', label='Train')
        ax_acc.plot(epochs, h["test_acc"], 'r-', label='Test')
        ax_acc.set_title(f'Acc (held out: {person})', fontsize=10)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lopo_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5))
    for i, (person, result) in enumerate(fold_results.items()):
        ax = axes[i] if num_folds > 1 else axes
        cm = result["confusion_matrix"]
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Held out: {person}\nAcc: {result["accuracy"]:.1%}', fontsize=10)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        short_labels = [c[:6] for c in CLASSES]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)

        for r in range(num_classes):
            for c_idx in range(num_classes):
                val = cm_norm[r, c_idx]
                color = 'white' if val > 0.5 else 'black'
                ax.text(c_idx, r, f'{val:.0%}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lopo_confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    persons = list(fold_results.keys())
    accs = [fold_results[p]["accuracy"] for p in persons]
    bars = ax.bar(range(len(persons)), accs, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    ax.set_xticks(range(len(persons)))
    ax.set_xticklabels(persons, fontsize=9)
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'LOPO Cross-Validation (Mean: {np.mean(accs):.1%}, Std: {np.std(accs):.1%})')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=np.mean(accs), color='gray', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(accs):.1%}')
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lopo_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to {save_dir}/")


def main():
    print("=" * 65)
    print("  CG4002 B02 - Leave-One-Person-Out Cross-Validation")
    print("=" * 65)

    print("\n  Loading data by person...")
    person_data = load_all_by_person()

    persons = sorted(person_data.keys())
    print(f"\n  {len(persons)} people found: {persons}")

    fold_results = {}
    all_preds = []
    all_labels = []

    for fold_idx, held_out in enumerate(persons):
        print(f"\n  {'='*55}")
        print(f"  Fold {fold_idx+1}/{len(persons)}: Held out = {held_out}")
        print(f"  {'='*55}")

        X_te, y_te = person_data[held_out]
        X_tr_parts, y_tr_parts = [], []
        for p in persons:
            if p != held_out:
                X_tr_parts.append(person_data[p][0])
                y_tr_parts.append(person_data[p][1])

        X_tr = np.concatenate(X_tr_parts, axis=0)
        y_tr = np.concatenate(y_tr_parts, axis=0)

        print(f"    Train: {len(X_tr)} windows from {len(persons)-1} people")
        print(f"    Test:  {len(X_te)} windows from {held_out}")

        test_classes = {CLASSES[i]: int(np.sum(y_te == i)) for i in range(len(CLASSES)) if np.sum(y_te == i) > 0}
        print(f"    Test classes: {test_classes}")

        best_acc, preds, labels, history = train_fold(X_tr, y_tr, X_te, y_te, held_out)
        cm = compute_confusion_matrix(labels, preds, len(CLASSES))

        fold_results[held_out] = {
            "accuracy": best_acc,
            "confusion_matrix": cm,
            "history": history,
        }
        all_preds.extend(preds)
        all_labels.extend(labels)

        print(f"    Best test accuracy: {best_acc:.1%}")

        for i, cls in enumerate(CLASSES):
            mask = labels == i
            if mask.sum() > 0:
                cls_acc = (preds[mask] == i).mean()
                print(f"      {cls:15s}: {cls_acc:.1%} ({mask.sum()} samples)")

    print(f"\n  {'='*55}")
    print(f"  LOPO CROSS-VALIDATION SUMMARY")
    print(f"  {'='*55}")

    accs = [fold_results[p]["accuracy"] for p in persons]
    for p in persons:
        print(f"    {p:15s}: {fold_results[p]['accuracy']:.1%}")
    print(f"    {'─'*30}")
    print(f"    {'Mean':15s}: {np.mean(accs):.1%}")
    print(f"    {'Std':15s}: {np.std(accs):.1%}")
    print(f"    {'Min':15s}: {np.min(accs):.1%}")
    print(f"    {'Max':15s}: {np.max(accs):.1%}")

    overall_cm = compute_confusion_matrix(np.array(all_labels), np.array(all_preds), len(CLASSES))
    overall_correct = sum(overall_cm[i][i] for i in range(len(CLASSES)))
    overall_total = overall_cm.sum()
    print(f"\n    Overall accuracy (all folds): {overall_correct/overall_total:.1%}")

    save_dir = "models/lopo"
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "method": "Leave-One-Person-Out Cross-Validation",
        "folds": {},
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "overall_accuracy": float(overall_correct / overall_total),
        "overall_confusion_matrix": overall_cm.tolist(),
    }
    for p in persons:
        results["folds"][p] = {
            "accuracy": fold_results[p]["accuracy"],
            "confusion_matrix": fold_results[p]["confusion_matrix"].tolist(),
        }

    with open(os.path.join(save_dir, "lopo_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_lopo_results(fold_results, save_dir)

    print(f"\n  Results saved to {save_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
