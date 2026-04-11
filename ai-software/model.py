"""
CG4002 B02 - 1D-CNN Model Definition (7 Classes)
=================================================
Architecture:

  Input:  (batch, 20, 12)  — 20 timesteps x 12 features
          3 sensors (arm, chest, thigh) x 4 channels (avm, ax, ay, az)
          Sampled at ~8 Hz from Real_Training_Data/Previous

  Conv1D       32 filters, kernel=9, padding='same'
  BatchNorm1d  32
  ReLU
  MaxPool1d    pool=2, stride=2            -> 10 timesteps
  Conv1D       64 filters, kernel=5, padding='same'
  BatchNorm1d  64
  ReLU
  GlobalAvgPool                             -> (batch, 64)
  Dense        64 -> 32, ReLU, Dropout(0.3)
  Dense        32 -> 7  (output logits)

  Output: (batch, 7)
"""

import torch
import torch.nn as nn


class ExerciseCNN(nn.Module):
    """1D CNN for IMU-based exercise classification."""

    def __init__(self, num_features=12, num_classes=7):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        self.bn1   = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.bn2   = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1   = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, window_size, num_features)
        x = x.permute(0, 2, 1)              # (batch, num_features, window_size)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)          # (batch, 64)
        x = self.relu3(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CLASSES = ["high_knees", "lunge", "squat", "overhead_arm", "push_up", "sit_up", "unknown"]


if __name__ == "__main__":
    m = ExerciseCNN(num_features=12, num_classes=7)
    x = torch.randn(4, 20, 12)
    print(m(x).shape)
    print(f"Params: {m.count_params():,}")
