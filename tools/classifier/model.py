"""Small CNN for one square crop -> 13 classes (~290K parameters).

Input (N, 1, 56, 48) float32, normalised by the board's mean and std
(kibitz_tools/crops.py). Output (N, 13) logits, classes as kibitz_tools/labels.py.
"""

import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    # Two 3x3 convs then a 2x2 max-pool: halves the resolution.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SquareNet(nn.Module):
    def __init__(self, num_classes: int = 13):
        super().__init__()
        # 56x48 -> 28x24 -> 14x12 -> 7x6, then global average pool.
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


def load(path) -> tuple[SquareNet, dict]:
    """Load a checkpoint written by train.py. Returns (model in eval mode, metadata)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SquareNet(len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck
