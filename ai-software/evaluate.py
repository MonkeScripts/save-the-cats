"""
CG4002 B02 - Step 4: Model Evaluation (7 Classes)
===================================================
Covers Task 5 demo requirements:
  - Confusion matrix (counts + normalized)
  - Classification report (precision, recall, F1 per class)
  - Per-class accuracy bar chart
  - Inference timing on CPU (baseline for FPGA comparison later)
  - Model validation summary

Usage:
  cd cg4002-ai-demo
  python software/evaluate.py
"""

import torch
import numpy as np
import os
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import ExerciseCNN

CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES = len(CLASSES)


def load_model_and_data():
    """Load trained model and normalized test data."""
    mean = np.load("models/norm_mean.npy")
    std  = np.load("models/norm_std.npy")

    X_test = np.load("data/test/X.npy")
    y_test = np.load("data/test/y.npy")
    X_test = ((X_test - mean) / std).astype(np.float32)

    model = ExerciseCNN(num_features=12, num_classes=7)
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.eval()

    print(f"  Model loaded: {model.count_params():,} parameters")
    print(f"  Test data:    {X_test.shape[0]} samples, {NUM_CLASSES} classes")
    return model, X_test, y_test


def run_inference(model, X_test):
    """Run inference, return predictions and timing stats."""
    X_tensor = torch.from_numpy(X_test)

    with torch.no_grad():
        _ = model(X_tensor[:1])

    start = time.time()
    with torch.no_grad():
        logits = model(X_tensor)
    batch_time = time.time() - start

    predictions = torch.argmax(logits, dim=1).numpy()
    confidences = torch.softmax(logits, dim=1).numpy()

    single_times = []
    for i in range(min(100, len(X_test))):
        t0 = time.time()
        with torch.no_grad():
            _ = model(X_tensor[i:i+1])
        single_times.append(time.time() - t0)

    avg_single_ms = np.mean(single_times) * 1000
    std_single_ms = np.std(single_times) * 1000

    print(f"\n  Inference timing (CPU - PyTorch):")
    print(f"    Batch ({len(X_test)} samples):   {batch_time*1000:.1f} ms total")
    print(f"    Single sample avg:      {avg_single_ms:.2f} ± {std_single_ms:.2f} ms")
    print(f"    Throughput:             {len(X_test)/batch_time:.0f} inferences/sec")

    return predictions, confidences, avg_single_ms


def compute_confusion_matrix(y_true, y_pred, n_classes):
    """Compute confusion matrix manually."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm):
    """Compute precision, recall, F1 per class from confusion matrix."""
    metrics = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = cm[i, :].sum()

        metrics[i] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
    return metrics


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix: raw counts and normalized side by side."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax1,
                cbar_kws={'label': 'Count'}, annot_kws={"size": 13})
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.tick_params(axis='x', rotation=30)
    ax1.tick_params(axis='y', rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax2,
                cbar_kws={'label': 'Proportion'}, annot_kws={"size": 13})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax2.tick_params(axis='x', rotation=30)
    ax2.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_per_class_accuracy(cm, save_path):
    """Bar chart of per-class accuracy."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4', '#795548']
    bars = ax.bar(CLASSES, per_class_acc, color=colors, edgecolor='white', linewidth=1.5)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Classification Accuracy', fontsize=14)
    mean_acc = np.mean(per_class_acc)
    ax.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {mean_acc:.1%}')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class accuracy saved: {save_path}")


def main():
    print("=" * 65)
    print("  CG4002 B02 — Step 4: Model Evaluation")
    print("=" * 65)

    print("\n  Loading model & data...")
    model, X_test, y_test = load_model_and_data()

    print("\n  Running inference...")
    y_pred, confidences, avg_latency_ms = run_inference(model, X_test)

    cm = compute_confusion_matrix(y_test, y_pred, NUM_CLASSES)
    overall_acc = cm.diagonal().sum() / cm.sum()

    print(f"\n  Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")

    metrics = compute_metrics(cm)

    macro_p = np.mean([metrics[i]['precision'] for i in range(NUM_CLASSES)])
    macro_r = np.mean([metrics[i]['recall'] for i in range(NUM_CLASSES)])
    macro_f = np.mean([metrics[i]['f1'] for i in range(NUM_CLASSES)])
    total_support = sum(metrics[i]['support'] for i in range(NUM_CLASSES))

    rows = [[cls, f"{metrics[i]['precision']:.4f}", f"{metrics[i]['recall']:.4f}",
             f"{metrics[i]['f1']:.4f}", metrics[i]['support']]
            for i, cls in enumerate(CLASSES)]
    rows += [
        ["macro avg", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f:.4f}", total_support],
        ["accuracy",  "",               "",               f"{overall_acc:.4f}", total_support],
    ]
    print(f"\n  Classification Report:")
    print(tabulate(rows, headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
                   tablefmt="simple"))

    print(f"\n  Generating plots...")
    os.makedirs("models", exist_ok=True)
    plot_confusion_matrix(cm, "models/confusion_matrix.png")
    plot_per_class_accuracy(cm, "models/per_class_accuracy.png")

    results = {
        "overall_accuracy": float(overall_acc),
        "avg_inference_ms_cpu": float(avg_latency_ms),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASSES[i]: {
                "precision": float(metrics[i]["precision"]),
                "recall": float(metrics[i]["recall"]),
                "f1": float(metrics[i]["f1"]),
                "support": metrics[i]["support"],
                "accuracy": float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0,
            }
            for i in range(NUM_CLASSES)
        },
        "macro_avg": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f),
        },
    }
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: models/evaluation_results.json")

    print(f"\n  Evaluation Summary:")
    print(tabulate([
        ["Overall Accuracy",  f"{overall_acc*100:.1f}%"],
        ["Macro Precision",   f"{macro_p:.4f}"],
        ["Macro Recall",      f"{macro_r:.4f}"],
        ["Macro F1-Score",    f"{macro_f:.4f}"],
        ["CPU Inference",     f"{avg_latency_ms:.2f} ms/sample"],
        ["Test Samples",      total_support],
        ["Validation Method", "Three-way split (train/val/test)"],
    ], tablefmt="simple"))
    print()
    print("  ✓ Step 4 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
