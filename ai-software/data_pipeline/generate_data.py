"""
CG4002 B02 - Step 1: Dummy IMU Data Generator (7 Classes)
==========================================================
Generates synthetic IMU data for 7 classes:
  0 = high_knees
  1 = pushup
  2 = situp
  3 = lunge
  4 = squat
  5 = overhead_hold
  6 = unknown  (unrecognised / wrong movement)

Hardware setup (from design report):
  3 IMUs (chest, wrist, thigh) x 6 axes (ax, ay, az, gx, gy, gz) = 18 features
  Sliding window: 128 samples at 50Hz = 2.56 seconds

Usage:
  cd cg4002-ai-demo
  python software/generate_data.py
"""

import numpy as np
import os
import json

np.random.seed(42)

NUM_SENSORS = 3             # chest, wrist, thigh
AXES_PER_SENSOR = 6         # ax, ay, az, gx, gy, gz
NUM_FEATURES = NUM_SENSORS * AXES_PER_SENSOR   # 18
WINDOW_SIZE = 128           # ~2.56s at 50Hz
SAMPLE_RATE = 50            # Hz

CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]
NUM_CLASSES = len(CLASSES)

TRAIN_PER_CLASS = 200       # 1400 total train
VAL_PER_CLASS  = 50         #  350 total val  (used during training for model selection)
TEST_PER_CLASS = 50         #  350 total test (held out — never seen during training)


def _gen_high_knees():
    """
    High knees signature:
    - Thigh: rapid, high-amplitude vertical oscillation (knees pumping up)
    - Chest: moderate vertical bounce from running-in-place
    - Wrist: arm swing counter to legs
    Fast frequency (~2-3 Hz) distinguishes from slower exercises.
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    freq = np.random.uniform(2.0, 3.0)

    # Chest [0:6] — vertical bounce
    d[:, 0] = 0.3 * np.sin(2 * np.pi * freq * t)
    d[:, 1] = 1.8 * np.sin(2 * np.pi * freq * t)         # ay: strong bounce
    d[:, 2] = 0.2 * np.sin(2 * np.pi * freq * t)
    d[:, 3] = 0.6 * np.cos(2 * np.pi * freq * t)
    d[:, 4] = 0.4 * np.sin(2 * np.pi * freq * 0.5 * t)
    d[:, 5] = 0.3 * np.cos(2 * np.pi * freq * t)

    # Wrist [6:12] — arm swing (opposite phase)
    d[:, 6]  = 1.0 * np.sin(2 * np.pi * freq * t + np.pi)
    d[:, 7]  = 1.2 * np.sin(2 * np.pi * freq * t)
    d[:, 8]  = 0.3 * np.sin(2 * np.pi * freq * t)
    d[:, 9]  = 0.8 * np.cos(2 * np.pi * freq * t)
    d[:, 10] = 0.3 * np.cos(2 * np.pi * freq * t)
    d[:, 11] = 0.4 * np.cos(2 * np.pi * freq * t)

    # Thigh [12:18] — dominant: rapid high-amplitude knee lifts
    d[:, 12] = 0.5 * np.sin(2 * np.pi * freq * t)
    d[:, 13] = 3.0 * np.sin(2 * np.pi * freq * t)        # ay: very strong vertical
    d[:, 14] = 0.3 * np.sin(2 * np.pi * freq * t)
    d[:, 15] = 2.5 * np.cos(2 * np.pi * freq * t)        # gx: rapid hip flexion
    d[:, 16] = 0.4 * np.cos(2 * np.pi * freq * t)
    d[:, 17] = 1.0 * np.cos(2 * np.pi * freq * t)
    return d


def _gen_pushup():
    """
    Push-up signature:
    - Chest: large vertical oscillation while prone (gravity on ax)
    - Wrist: fixed on ground, pressure/angle changes
    - Thigh: horizontal, mostly steady
    Body is prone so gravity shifts to ax (~9.5).
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    freq = np.random.uniform(0.5, 1.0)

    # Chest [0:6] — prone, large up-down
    d[:, 0] = 9.5 + 0.3 * np.sin(2 * np.pi * freq * t)  # ax: gravity (face down)
    d[:, 1] = 2.0 * np.sin(2 * np.pi * freq * t)         # ay: push motion
    d[:, 2] = 0.2 * np.sin(2 * np.pi * freq * t)
    d[:, 3] = 1.5 * np.cos(2 * np.pi * freq * t)         # gx: chest pitch
    d[:, 4] = 0.1 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 5] = 0.3 * np.cos(2 * np.pi * freq * t)

    # Wrist [6:12] — fixed on ground
    d[:, 6]  = 9.3                                         # ax: gravity (palm down)
    d[:, 7]  = 0.4 * np.sin(2 * np.pi * freq * t)
    d[:, 8]  = 0.1 * np.sin(2 * np.pi * freq * t)
    d[:, 9]  = 0.3 * np.cos(2 * np.pi * freq * t)
    d[:, 10] = 0.1 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 11] = 0.2 * np.cos(2 * np.pi * freq * t)

    # Thigh [12:18] — horizontal, mostly steady
    d[:, 12] = 9.4                                         # ax: gravity (prone)
    d[:, 13] = 0.5 * np.sin(2 * np.pi * freq * t)
    d[:, 14] = 0.1 * np.sin(2 * np.pi * freq * t)
    d[:, 15] = 0.4 * np.cos(2 * np.pi * freq * t)
    d[:, 16] = 0.1 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 17] = 0.2 * np.cos(2 * np.pi * freq * t)
    return d


def _gen_situp():
    """
    Sit-up signature:
    - Chest: large pitch rotation (flat ↔ upright), gravity rotates ax↔ay
    - Wrist: follows torso (hands behind head)
    - Thigh: anchored on ground, mostly still
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    freq = np.random.uniform(0.5, 0.9)

    # Torso angle oscillates between lying (0) and sitting (pi/2)
    angle = np.pi / 2 * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t))

    # Chest [0:6] — gravity rotates as torso pitches
    d[:, 0] = 9.8 * np.cos(angle)                         # ax: lying component
    d[:, 1] = 9.8 * np.sin(angle)                         # ay: sitting component
    d[:, 2] = 0.2 * np.sin(2 * np.pi * freq * t)
    d[:, 3] = 2.0 * np.cos(2 * np.pi * freq * t)         # gx: large pitch rotation
    d[:, 4] = 0.2 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 5] = 0.3 * np.cos(2 * np.pi * freq * t)

    # Wrist [6:12] — follows torso (hands behind head)
    d[:, 6]  = 9.8 * np.cos(angle) * 0.9
    d[:, 7]  = 9.8 * np.sin(angle) * 0.9
    d[:, 8]  = 0.15 * np.sin(2 * np.pi * freq * t)
    d[:, 9]  = 1.5 * np.cos(2 * np.pi * freq * t)
    d[:, 10] = 0.2 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 11] = 0.2 * np.cos(2 * np.pi * freq * t)

    # Thigh [12:18] — anchored, mostly still
    d[:, 12] = 9.5                                         # ax: gravity (flat)
    d[:, 13] = 0.3 * np.sin(2 * np.pi * freq * t)
    d[:, 14] = 0.1 * np.sin(2 * np.pi * freq * t)
    d[:, 15] = 0.4 * np.cos(2 * np.pi * freq * t)
    d[:, 16] = 0.1 * np.random.normal(0, 0.05, WINDOW_SIZE)
    d[:, 17] = 0.15 * np.cos(2 * np.pi * freq * t)
    return d


def _gen_lunge():
    """
    Lunge signature:
    - Thigh: strong forward-back asymmetric motion
    - Chest: forward lean
    - Wrist: slight counter-swing
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    freq = np.random.uniform(0.4, 0.8)

    # Chest: forward tilt
    d[:, 0] = 1.2 * np.sin(2 * np.pi * freq * t)
    d[:, 1] = 1.0 * np.sin(2 * np.pi * freq * t)
    d[:, 2] = 0.3 * np.sin(2 * np.pi * freq * t)
    d[:, 3] = 1.0 * np.cos(2 * np.pi * freq * t)
    d[:, 4] = 0.3 * np.cos(2 * np.pi * freq * t)
    d[:, 5] = 0.2 * np.random.normal(0, 0.1, WINDOW_SIZE)

    # Wrist: counter-swing
    d[:, 6] = 0.6 * np.sin(2 * np.pi * freq * t + 0.5)
    d[:, 7:12] = np.random.normal(0, 0.15, (WINDOW_SIZE, 5))

    # Thigh: dominant forward-back
    d[:, 12] = 1.8 * np.sin(2 * np.pi * freq * t)
    d[:, 13] = 1.5 * np.sin(2 * np.pi * freq * t)
    d[:, 14] = 0.4 * np.sin(2 * np.pi * freq * t)
    d[:, 15] = 1.8 * np.cos(2 * np.pi * freq * t)
    d[:, 16] = 0.8 * np.cos(2 * np.pi * freq * t)
    d[:, 17] = 0.3 * np.random.normal(0, 0.1, WINDOW_SIZE)
    return d


def _gen_squat():
    """
    Squat signature:
    - Thigh: large vertical oscillation (femur angle change)
    - Chest: moderate vertical movement (torso dips)
    - Wrist: minimal (arms stay still)
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    freq = np.random.uniform(0.6, 1.0)

    # Chest
    d[:, 1] = 1.5 * np.sin(2 * np.pi * freq * t)
    d[:, 3] = 0.5 * np.cos(2 * np.pi * freq * t)
    d[:, 5] = 0.8 * np.cos(2 * np.pi * freq * t)

    # Wrist — nearly flat
    d[:, 6:12] = np.random.normal(0, 0.08, (WINDOW_SIZE, 6))

    # Thigh — dominant
    d[:, 13] = 2.5 * np.sin(2 * np.pi * freq * t)
    d[:, 15] = 2.0 * np.cos(2 * np.pi * freq * t)
    d[:, 17] = 1.5 * np.cos(2 * np.pi * freq * t)
    return d


def _gen_overhead():
    """
    Overhead arm hold signature:
    - Wrist: elevated (gravity ~9.5 on ay), fatigue tremor
    - Chest: upright, steady (gravity ~9.8 on ay)
    - Thigh: steady standing
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))

    # Chest: upright
    d[:, 1] = 9.8
    d[:, 0:6] += np.random.normal(0, 0.08, (WINDOW_SIZE, 6))

    # Wrist: held high — key discriminator
    d[:, 7]  = 9.5
    d[:, 9]  = 0.3 * np.sin(2 * np.pi * 3.0 * t)
    d[:, 10] = 0.2 * np.sin(2 * np.pi * 3.5 * t)
    d[:, 6:12] += np.random.normal(0, 0.12, (WINDOW_SIZE, 6))

    # Thigh: standing still
    d[:, 13] = 9.8
    d[:, 12:18] += np.random.normal(0, 0.05, (WINDOW_SIZE, 6))
    return d


def _gen_unknown():
    """
    Unknown / unrecognised movement signature.
    Randomly selects one of four sub-types so the model sees a wide variety
    of non-exercise signals rather than a single stereotype:

      0 - Random walk (Brownian motion): cumulative random steps, no periodic
          structure, random amplitude per channel.
      1 - Sedentary / near-static: device nearly still (e.g. sitting idle),
          only gravity bias + very low-amplitude sensor noise.
      2 - Mixed-frequency chaos: random superposition of 3-6 sine waves at
          arbitrary frequencies, amplitudes, and phases per channel — does not
          resemble any single exercise frequency.
      3 - Transitional / incomplete: first half looks like a slow oscillation
          (partial attempt at an exercise), second half abruptly shifts to a
          different frequency, mimicking an interrupted or wrong movement.
    """
    t = np.linspace(0, 2.56, WINDOW_SIZE)
    d = np.zeros((WINDOW_SIZE, NUM_FEATURES))
    sub_type = np.random.randint(0, 4)

    if sub_type == 0:
        # Random walk: no periodic structure, drifts unpredictably
        for ch in range(NUM_FEATURES):
            amp = np.random.uniform(0.2, 1.5)
            steps = np.random.normal(0, amp * 0.15, WINDOW_SIZE)
            d[:, ch] = np.cumsum(steps)
            d[:, ch] -= d[:, ch].mean()   # centre around zero

    elif sub_type == 1:
        # Sedentary: gravity on ay (upright standing/sitting), minimal motion
        d[:, 1]  = 9.8   # chest ay — gravity
        d[:, 7]  = 9.8   # wrist ay — gravity
        d[:, 13] = 9.8   # thigh ay — gravity
        d += np.random.normal(0, 0.04, d.shape)

    elif sub_type == 2:
        # Mixed-frequency: random superposition, no dominant exercise frequency
        for ch in range(NUM_FEATURES):
            n_components = np.random.randint(3, 7)
            for _ in range(n_components):
                freq  = np.random.uniform(0.1, 5.0)
                amp   = np.random.uniform(0.1, 1.2)
                phase = np.random.uniform(0, 2 * np.pi)
                d[:, ch] += amp * np.sin(2 * np.pi * freq * t + phase)

    else:
        # Transitional / interrupted: abrupt mid-window frequency change
        split = WINDOW_SIZE // 2
        freq1 = np.random.uniform(0.4, 1.2)   # slow first half
        freq2 = np.random.uniform(1.8, 4.5)   # fast second half
        for ch in range(NUM_FEATURES):
            amp1 = np.random.uniform(0.3, 2.0)
            amp2 = np.random.uniform(0.3, 2.0)
            d[:split, ch] = amp1 * np.sin(2 * np.pi * freq1 * t[:split])
            d[split:, ch] = amp2 * np.sin(2 * np.pi * freq2 * t[split:])

    return d


GENERATORS = [
    _gen_high_knees,    # 0
    _gen_pushup,        # 1
    _gen_situp,         # 2
    _gen_lunge,         # 3
    _gen_squat,         # 4
    _gen_overhead,      # 5
    _gen_unknown,       # 6
]


def _add_variation(data):
    """Add realistic per-sample variation: amplitude jitter, noise, time shift."""
    amp   = np.random.uniform(0.85, 1.15)
    noise = np.random.normal(0, 0.12, data.shape)
    shift = np.random.randint(-5, 6)
    return np.roll(data * amp + noise, shift, axis=0)


def generate_dataset(samples_per_class):
    """Generate (X, y) arrays with balanced classes."""
    X_list, y_list = [], []
    for cls_idx, gen_fn in enumerate(GENERATORS):
        for _ in range(samples_per_class):
            X_list.append(_add_variation(gen_fn()))
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def main():
    print("=" * 60)
    print("  CG4002 B02 — Step 1: Dummy IMU Data Generator")
    print("=" * 60)
    print(f"  Classes ({NUM_CLASSES}):")
    for i, c in enumerate(CLASSES):
        print(f"    {i} = {c}")
    print(f"  Features:   {NUM_FEATURES} (3 IMUs × 6 axes)")
    print(f"  Window:     {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz)")
    print(f"  Train/test: {TRAIN_PER_CLASS}/class  {TEST_PER_CLASS}/class")

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val",   exist_ok=True)
    os.makedirs("data/test",  exist_ok=True)

    print("\n  Generating training set...")
    X_tr, y_tr = generate_dataset(TRAIN_PER_CLASS)
    np.save("data/train/X.npy", X_tr)
    np.save("data/train/y.npy", y_tr)
    print(f"    X_train: {X_tr.shape}   y_train: {y_tr.shape}")
    print(f"    Class counts: {dict(zip(CLASSES, np.bincount(y_tr)))}")

    print("\n  Generating validation set...")
    X_va, y_va = generate_dataset(VAL_PER_CLASS)
    np.save("data/val/X.npy", X_va)
    np.save("data/val/y.npy", y_va)
    print(f"    X_val:   {X_va.shape}   y_val:   {y_va.shape}")
    print(f"    Class counts: {dict(zip(CLASSES, np.bincount(y_va)))}")

    print("\n  Generating test set...")
    X_te, y_te = generate_dataset(TEST_PER_CLASS)
    np.save("data/test/X.npy", X_te)
    np.save("data/test/y.npy", y_te)
    print(f"    X_test:  {X_te.shape}   y_test:  {y_te.shape}")
    print(f"    Class counts: {dict(zip(CLASSES, np.bincount(y_te)))}")

    meta = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "num_features": NUM_FEATURES,
        "window_size": WINDOW_SIZE,
        "sample_rate": SAMPLE_RATE,
        "sensor_locations": ["chest", "wrist", "thigh"],
        "axes_per_sensor": ["ax", "ay", "az", "gx", "gy", "gz"],
        "train_total": int(len(X_tr)),
        "val_total": int(len(X_va)),
        "test_total": int(len(X_te)),
    }
    with open("data/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Feature value ranges (training):")
    print(f"    Min: {X_tr.min():.3f}   Max: {X_tr.max():.3f}   Mean: {X_tr.mean():.3f}")

    print("\n  ✓ Step 1 complete! Files saved in data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
