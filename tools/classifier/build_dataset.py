"""Run dirs -> labelled square crops (.npz) for the piece classifier.

Only "commit" frames are used: their `fen` is the position on the board.
Each crop is stored with 4 px of jitter room on every side; `train.py` cuts a
random window from it, `eval.py` the centre. Geometry: kibitz_tools/crops.py.

    python classifier/build_dataset.py data/run1 data/run2 -o data/crops.npz

Synthetic data: make run dirs from a bench dump first (`python -m kibitz_tools.fake_run`).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kibitz_tools.crops import CROP_H, CROP_W, board_moments, oriented_warp, square_crops  # noqa: E402
from kibitz_tools.labels import CLASSES, fen_labels  # noqa: E402
from kibitz_tools.rundir import load_run  # noqa: E402

JITTER = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    crops, labels, squares, games, frames, means, stds = [], [], [], [], [], [], []
    for path in args.runs:
        run = load_run(path)
        # A run dir is one game (one `kibitz run`). Use its path as the game id
        # so train.py can split by game.
        game = str(path.resolve())
        n = 0
        for f in run.commits():
            w = oriented_warp(f.gray(), f.corners, f.my_color)
            if w is None:
                continue  # degenerate corners
            mean, std = board_moments(w)
            crops.append(square_crops(w, f.my_color, extra=JITTER))
            labels.append(fen_labels(f.fen))
            squares.append(np.arange(64, dtype=np.uint8))
            games += [game] * 64
            frames += [f.ts_ms] * 64
            means += [mean] * 64
            stds += [std] * 64
            n += 1
        print(f"{path}: {n} commit frames")
    if not crops:
        sys.exit("no labelled frames")

    labels = np.concatenate(labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        crops=np.concatenate(crops),  # (N, CROP_H + 2*JITTER, CROP_W + 2*JITTER) uint8
        labels=labels,  # (N,) uint8, index into CLASSES
        square=np.concatenate(squares),  # (N,) uint8, a1 = 0
        game=np.array(games),
        frame_ts=np.array(frames, np.int64),
        board_mean=np.array(means, np.float32),
        board_std=np.array(stds, np.float32),
        jitter=JITTER,
        crop_hw=np.array([CROP_H, CROP_W]),
    )
    counts = np.bincount(labels, minlength=len(CLASSES))
    print(f"wrote {args.out}: {labels.size} crops from {len(set(games))} games")
    print("  " + "  ".join(f"{c}={k}" for c, k in zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
