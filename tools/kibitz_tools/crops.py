"""Square crops for the piece classifier. The geometry here is the model
contract (classifier/MODEL_CONTRACT.md); change both together.

1. Warp the frame at `SQ_PX` = 32 with a `PAD` = 24 px border (`warp_board(...,
   pad_px=PAD)`), so edge squares have real context around them.
2. Orient to the player's view: for my_color "black", rotate the padded warp
   180 degrees. The player's own side is then at the bottom, and a tall piece
   leans upward in the image (its top projects toward the far side).
3. Crop each square with `SIDE` px of context left, right and below, and `TOP`
   px above (more above, because that is where pieces lean): `CROP_W` x `CROP_H`.
4. Normalise with the mean and std of the unpadded 8x8 board (std floored at 1),
   the same global moments as `square_stats` in features.rs.
"""

import numpy as np

from .board import SQ_PX, square_to_grid, warp_board

PAD = 24  # >= TOP + the 4 px training jitter in build_dataset.py
SIDE = 8
TOP = 16
CROP_W = SQ_PX + 2 * SIDE  # 48
CROP_H = SQ_PX + TOP + SIDE  # 56


def oriented_warp(gray: np.ndarray, corners, my_color: str, pad: int = PAD) -> np.ndarray | None:
    """Padded warp, rotated 180 degrees for the Black player (see module docs)."""
    w = warp_board(gray, corners, SQ_PX, pad_px=pad)
    if w is None:
        return None
    return w[::-1, ::-1] if my_color == "black" else w


def board_moments(warped: np.ndarray, pad: int = PAD) -> tuple[float, float]:
    """Mean and std (floored at 1) of the board area of a padded warp."""
    n = 8 * SQ_PX
    b = warped[pad : pad + n, pad : pad + n].astype(np.float64)
    return float(b.mean()), float(max(b.std(), 1.0))


def square_origin(square: int, my_color: str, pad: int = PAD) -> tuple[int, int]:
    """Top-left (x, y) of a square in the oriented padded warp."""
    row, col = square_to_grid(square)
    if my_color == "black":
        row, col = 7 - row, 7 - col
    return pad + col * SQ_PX, pad + row * SQ_PX


def square_crops(warped: np.ndarray, my_color: str, extra: int = 0, pad: int = PAD) -> np.ndarray:
    """(64, CROP_H + 2*extra, CROP_W + 2*extra) uint8 crops in square order a1..h8.
    `extra` adds jitter room on every side (training stores crops with extra = 4
    and cuts a random CROP_H x CROP_W window). Needs pad >= TOP + extra."""
    out = np.zeros((64, CROP_H + 2 * extra, CROP_W + 2 * extra), np.uint8)
    for s in range(64):
        x, y = square_origin(s, my_color, pad)
        out[s] = warped[y - TOP - extra : y + SQ_PX + SIDE + extra, x - SIDE - extra : x + SQ_PX + SIDE + extra]
    return out


def normalise(crops: np.ndarray, mean, std) -> np.ndarray:
    """uint8 crops -> float32 model input units. `mean`/`std` broadcast per crop."""
    mean = np.asarray(mean, np.float32).reshape(-1, *([1] * (crops.ndim - 1)))
    std = np.asarray(std, np.float32).reshape(-1, *([1] * (crops.ndim - 1)))
    return (crops.astype(np.float32) - mean) / std
