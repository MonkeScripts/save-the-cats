"""
CG4002 B02 - Feature utilities for IMU windows
===============================================
Handles raw accelerometer data from 3 sensors (arm, chest, thigh),
each providing 4 channels: avm, ax, ay, az.

Sensor layout (SENSOR_ORDER = ["arm", "chest", "thigh"]):
  arm:   channels 0-3   (avm=0, ax=1, ay=2, az=3)
  chest: channels 4-7   (avm=4, ax=5, ay=6, az=7)
  thigh: channels 8-11  (avm=8, ax=9, ay=10, az=11)

Total raw features per timestep: 12 (4 per sensor x 3 sensors)
No hand-crafted features — the CNN learns representations directly.
"""

import numpy as np

SAMPLE_RATE = 8   # Hz (data collected at ~8 Hz from Previous folder)

SENSOR_ORDER = ["arm", "chest", "thigh"]
FEATURES_PER_SENSOR = ["avm", "ax", "ay", "az"]
NUM_FEATURES = len(SENSOR_ORDER) * len(FEATURES_PER_SENSOR)  # 12

# Channel indices per sensor
ARM_CHANNELS   = list(range(0, 4))   # 0-3
CHEST_CHANNELS = list(range(4, 8))   # 4-7
THIGH_CHANNELS = list(range(8, 12))  # 8-11

# Accelerometer axis indices within each sensor block (ax=1, ay=2, az=3 offset)
ACCEL_AXIS_OFFSETS = [1, 2, 3]  # ax, ay, az within each 4-channel block


def remove_gravity(X):
    """
    Remove gravity bias from accelerometer axes (ax, ay, az) via window-mean subtraction.

    Gravity appears as a slow DC offset on the acceleration axes, varying with
    sensor orientation. Subtracting the window mean reduces orientation-dependent
    bias across people.

    Args:
        X: numpy array, shape (..., T, 12)

    Returns:
        numpy array, same shape, accel axes gravity-removed
    """
    X = X.copy()
    # Indices of ax, ay, az for all 3 sensors: 1,2,3, 5,6,7, 9,10,11
    accel_indices = [offset + base for base in [0, 4, 8] for offset in ACCEL_AXIS_OFFSETS]
    X[..., accel_indices] -= X[..., accel_indices].mean(axis=-2, keepdims=True)
    return X


def augment_data(X, y, rng=None, n_copies=2,
                 scale_range=(0.85, 1.15),
                 noise_std=0.05,
                 time_shift_max=4):
    """
    Data augmentation for training to improve cross-person generalization.

    Three augmentations applied per copy:
      - Random scaling:  multiply each window by U(0.85, 1.15)
      - Additive noise:  add N(0, 0.05) Gaussian noise
      - Time shift:      roll window along time axis by up to ±4 samples

    Args:
        X: numpy array, shape (N, T, C)
        y: numpy array, shape (N,)
        rng: numpy random Generator
        n_copies: number of augmented copies per original
        scale_range: (min, max) uniform scale factor
        noise_std: std of additive Gaussian noise
        time_shift_max: max samples to shift along time axis

    Returns:
        X_aug, y_aug — concatenation of original + augmented windows
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N, T, C = X.shape
    copies_X = [X]
    copies_y = [y]

    for _ in range(n_copies):
        Xc = X.copy()
        scales = rng.uniform(scale_range[0], scale_range[1], size=(N, 1, 1)).astype(np.float32)
        Xc = Xc * scales
        Xc += rng.normal(0, noise_std, size=Xc.shape).astype(np.float32)
        shifts = rng.integers(-time_shift_max, time_shift_max + 1, size=N)
        for i, s in enumerate(shifts):
            if s != 0:
                Xc[i] = np.roll(Xc[i], s, axis=0)
        copies_X.append(Xc)
        copies_y.append(y)

    return np.concatenate(copies_X, axis=0), np.concatenate(copies_y, axis=0)
