"""Per-square diffs between consecutive commit frames, with ground truth.

At runtime the reference is the square stats of the frame the last move was
read from, and the next read diffs against it (cg-app/src/vision.rs). A
"commit" frame is exactly that frame, so diffing consecutive commit frames
replays what the runtime saw, except that the saved frames went through JPEG
(quality 80), which adds some noise the live pipeline doesn't have.
"""

from dataclasses import dataclass

import chess
import numpy as np

from .board import SQ_PX, square_diff, square_stats, warp_board
from .labels import changed_squares
from .rundir import FrameRecord, Run


@dataclass
class PairDiff:
    run: str
    before: FrameRecord
    after: FrameRecord
    move: chess.Move
    diff: np.ndarray  # (64,) f32, square_diff(before, after)
    changed: np.ndarray  # (64,) bool, squares whose occupant really changed


def pair_diffs(run: Run, sq_px: int = SQ_PX) -> list[PairDiff]:
    """Diffs of every consecutive commit pair one legal move apart."""
    out = []
    stats = {}  # each commit frame is the "after" of one pair and the "before" of the next

    def get(f: FrameRecord):
        if f.ts_ms not in stats:
            stats[f.ts_ms] = square_stats(warp_board(f.gray(), f.corners, sq_px), sq_px)
        return stats[f.ts_ms]

    for a, b, mv in run.commit_pairs():
        truth = np.zeros(64, bool)
        truth[changed_squares(a.fen, b.fen)] = True
        out.append(PairDiff(run.name, a, b, mv, square_diff(get(a), get(b)), truth))
    return out


def threshold(diff: np.ndarray, min_change: float, median_factor: float = 2.5) -> float:
    """Per-read change threshold, as in `excess_over_threshold` (infer.rs):
    max(min_change, 2.5 * median of the 64 diffs)."""
    s = np.sort(diff)
    return max(min_change, median_factor * (s[31] + s[32]) / 2)
