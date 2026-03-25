"""
CG4002 B02 - Step 2: 1D-CNN Model Definition (7 Classes)
=========================================================
Architecture (from design report Section 5.2.3):

  Input:  (batch, 128, 18)  — 128 timesteps x 18 features (3 IMUs x 6 axes)

  Layer 1: Conv1D       32 filters, kernel=9, padding='same', stride=1
           BatchNorm1d  32
           ReLU
  Layer 2: MaxPool1d    pool=2, stride=2            → 128 → 64 timesteps
  Layer 3: Conv1D       64 filters, kernel=5, padding='same', stride=1
           BatchNorm1d  64
           ReLU
  Layer 4: GlobalAvgPool  average across time dim   → (64,)
  Layer 5: Dense        64 → 32, ReLU
  Layer 6: Dense        32 → 7  (output logits)

  Output: (batch, 7)  — high_knees / pushup / situp / lunge / squat / overhead_hold / unknown

Usage:
  python software/model.py
"""

import torch
import torch.nn as nn


class ExerciseCNN(nn.Module):
    """1D Convolutional Neural Network for IMU-based exercise classification."""

    def __init__(self, num_features=12, num_classes=7):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=9,
            stride=1,
            padding=4   # 'same' padding = (kernel-1)//2
        )
        self.bn1   = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2   # 'same' padding
        )
        self.bn2   = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.gap = nn.AvgPool1d(kernel_size=10)

        self.fc1   = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        x = self.relu3(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CLASSES = ["high_knees", "pushup", "situp", "lunge", "squat", "overhead_hold", "unknown"]


def print_model_summary():
    from torchinfo import summary
    m = ExerciseCNN(num_features=12, num_classes=7)
    print(f"Output classes: {CLASSES}\n")
    summary(m, input_size=(1, 20, 12), col_names=["input_size", "output_size", "num_params"])


if __name__ == "__main__":
    print_model_summary()
