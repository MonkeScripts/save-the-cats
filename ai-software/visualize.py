"""
CG4002 B02 - Training Data Visualization
==========================================
Generates diagnostic plots for the real IMU training data.

Plots generated:
  1. IMU signals by class (all exercises, first person, 3 sensors)
  2. Cross-person comparison (same exercise across all people)
  3. t-SNE embedding of training windows (class separability)
  4. Feature correlation matrix (sensor/axis redundancy)

Usage:
  python visualize.py

All outputs saved to models/plots/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


DATA_DIR = "data/Real_Training_Data"
OUTPUT_DIR = "models/plots"

SENSOR_ORDER = ["arm", "chest", "thigh"]
FEATURES = ["ax", "ay", "az", "avm"]
CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]

FEATURE_LABELS = [f"{s}_{f}" for s in SENSOR_ORDER for f in FEATURES]

COLORS_FEAT = {"ax": "#e74c3c", "ay": "#2ecc71", "az": "#3498db", "avm": "#9b59b6"}
COLORS_CLASS = {
    "high_knees": "#e74c3c",
    "pushup": "#3498db",
    "situp": "#2ecc71",
    "lunge": "#f39c12",
    "squat": "#9b59b6",
    "overhead_hold": "#1abc9c",
    "unknown": "#95a5a6",
}

# Maps filename -> class for all persons
FILENAME_TO_CLASS = {
    "high-knee, 30s.csv": "high_knees",
    "push-up, 30s.csv": "pushup",
    "sit-up, 30s.csv": "situp",
    "lunges, 30s.csv": "lunge",
    "squats, 30s.csv": "squat",
    "overhead_arm, 30s.csv": "overhead_hold",
    "control-stationary, 30s.csv": "unknown",
    "v2high-knee, 30s.csv": "high_knees",
    "v2push-up, 30s.csv": "pushup",
    "v2sit-up, 30s.csv": "situp",
    "v2lunges, 30s.csv": "lunge",
    "v2squat, 30s.csv": "squat",
    "v2overhead_arm, 30s.csv": "overhead_hold",
    "v3high-knee.csv": "high_knees",
    "v3pushup.csv": "pushup",
    "v3situp.csv": "situp",
    "v3lunges.csv": "lunge",
    "v3squats.csv": "squat",
    "v3overhead_hold.csv": "overhead_hold",
    "v3control.csv": "unknown",
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


def parse_file(filepath):
    """Parse InfluxDB CSV, return {sensor: [list of payload dicts]}."""
    sensor_data = {s: [] for s in SENSOR_ORDER}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ",value,esp/" not in line:
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
            raw_json = line[json_start + 1:json_end + 1].replace('""', '"')
            try:
                sensor_data[sensor].append(json.loads(raw_json))
            except (json.JSONDecodeError, ValueError):
                pass
    return sensor_data


def load_all_data():
    """Load all CSV files, organized by person and class.

    Returns:
        dict: {person_name: {class_name: {sensor: [readings]}}}
    """
    all_data = {}
    for person_dir in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_dir)
        if not os.path.isdir(person_path) or person_dir.startswith("."):
            continue
        all_data[person_dir] = {}
        for fname in sorted(os.listdir(person_path)):
            if not fname.endswith(".csv"):
                continue
            cls = FILENAME_TO_CLASS.get(fname)
            if cls is None:
                continue
            fpath = os.path.join(person_path, fname)
            all_data[person_dir][cls] = parse_file(fpath)
    return all_data


def plot_signals_by_class(all_data):
    """Plot all 7 classes x 3 sensors for the first person."""
    person = list(all_data.keys())[0]
    data = all_data[person]

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(20, 28))
    fig.suptitle(f"IMU Sensor Data by Exercise Class ({person}, 30s)",
                 fontsize=16, fontweight='bold', y=0.995)

    for row, cls in enumerate(CLASSES):
        if cls not in data:
            continue
        for col, sensor in enumerate(SENSOR_ORDER):
            ax = axes[row][col]
            readings = data[cls][sensor]
            if not readings:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes)
                continue

            t = np.arange(len(readings)) / 8.0
            for feat in FEATURES:
                vals = [r[feat] for r in readings]
                lw = 1.5 if feat == "avm" else 1.0
                ls = '--' if feat == "avm" else '-'
                ax.plot(t, vals, color=COLORS_FEAT[feat], linewidth=lw,
                        linestyle=ls, label=feat, alpha=0.85)

            if row == 0:
                ax.set_title(f"{sensor.upper()}", fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{cls}\n(m/s^2)", fontsize=11, fontweight='bold')
            if row == len(CLASSES) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, t[-1])
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = os.path.join(OUTPUT_DIR, "1_signals_by_class.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1/4] Saved: {path}")


def plot_cross_person(all_data):
    """For each exercise, overlay the arm sensor signal from all people."""
    persons = list(all_data.keys())
    person_colors = plt.cm.tab10(np.linspace(0, 1, len(persons)))

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(22, 28))
    fig.suptitle("Cross-Person Comparison: Same Exercise, Different People",
                 fontsize=16, fontweight='bold', y=0.995)

    for row, cls in enumerate(CLASSES):
        for col, sensor in enumerate(SENSOR_ORDER):
            ax = axes[row][col]

            has_data = False
            for pidx, person in enumerate(persons):
                if cls not in all_data[person]:
                    continue
                readings = all_data[person][cls][sensor]
                if not readings:
                    continue
                has_data = True

                t = np.arange(len(readings)) / 8.0
                vals = [r["avm"] for r in readings]
                label_str = person.replace(" person", "").strip().capitalize()
                ax.plot(t, vals, color=person_colors[pidx], linewidth=1.2,
                        label=f"P{pidx+1} ({label_str})", alpha=0.8)

            if not has_data:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes)

            if row == 0:
                ax.set_title(f"{sensor.upper()} (avm)", fontsize=13,
                             fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{cls}\n(m/s^2)", fontsize=11, fontweight='bold')
            if row == len(CLASSES) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    path = os.path.join(OUTPUT_DIR, "2_cross_person.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2/4] Saved: {path}")


def plot_tsne():
    """t-SNE visualization of all training windows colored by class."""
    X_tr = np.load("data/train/X.npy")
    y_tr = np.load("data/train/y.npy")
    X_va = np.load("data/val/X.npy")
    y_va = np.load("data/val/y.npy")
    X_te = np.load("data/test/X.npy")
    y_te = np.load("data/test/y.npy")

    X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)

    X_flat = X_all.reshape(len(X_all), -1)

    X_flat = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Training Window Embeddings (all data)",
                 fontsize=16, fontweight='bold')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_all == cls_idx
        if not mask.any():
            continue
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS_CLASS[cls_name],
                    label=cls_name, alpha=0.7, s=40, edgecolors='white',
                    linewidth=0.3)
    ax1.set_title(f"PCA (explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                  fontsize=13)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.2)

    print("    Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_flat) - 1),
                random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_flat)

    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_all == cls_idx
        if not mask.any():
            continue
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=COLORS_CLASS[cls_name],
                    label=cls_name, alpha=0.7, s=40, edgecolors='white',
                    linewidth=0.3)
    ax2.set_title("t-SNE", fontsize=13)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_tsne_pca.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3/4] Saved: {path}")


def plot_correlation():
    """Correlation matrix of all 12 features across all training data."""
    X_tr = np.load("data/train/X.npy")

    X_flat = X_tr.reshape(-1, X_tr.shape[-1])

    corr = np.corrcoef(X_flat.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(len(FEATURE_LABELS)))
    ax.set_yticks(range(len(FEATURE_LABELS)))
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9)

    for i in range(len(FEATURE_LABELS)):
        for j in range(len(FEATURE_LABELS)):
            val = corr[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=7, color=color)

    for boundary in [4, 8]:
        ax.axhline(boundary - 0.5, color='black', linewidth=1.5)
        ax.axvline(boundary - 0.5, color='black', linewidth=1.5)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
    ax.set_title("Feature Correlation Matrix (12 features, training data)",
                 fontsize=14, fontweight='bold', pad=15)

    for idx, sensor in enumerate(SENSOR_ORDER):
        mid = idx * 4 + 1.5
        ax.text(mid, -1.2, sensor.upper(), ha='center', fontsize=10,
                fontweight='bold', color=COLORS_CLASS[list(COLORS_CLASS.keys())[idx]])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_feature_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [4/4] Saved: {path}")


def main():
    print("=" * 65)
    print("  CG4002 B02 - Training Data Visualization")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n  Loading all CSV data...")
    all_data = load_all_data()
    persons = list(all_data.keys())
    print(f"  Found {len(persons)} people: {', '.join(persons)}")

    print("\n  Generating plots...\n")

    plot_signals_by_class(all_data)
    plot_cross_person(all_data)
    plot_tsne()
    plot_correlation()

    print(f"\n  All plots saved to {OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
