"""
Shared CNN architecture used by both training and inference.
"""

import torch.nn as nn


class MNISTNet(nn.Module):
    """
    Small CNN for MNIST digit classification.
    Input: (B, 1, 28, 28)  Output: (B, 10) logits
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
