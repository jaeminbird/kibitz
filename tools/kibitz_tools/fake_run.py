"""Build fake run directories from `kibitz bench --dump` output, for smoke
tests while there is no real footage.

`bench --dump DIR --frames N` (crates/cg-app/src/bench.rs, `dump`) writes
PNGs only, no labels. The labels are recoverable because the dump is
deterministic: frames 10k .. 10k+9 show the position after k plies, rendered
with `synth::default_params(1280, 720)` corners shifted by 6k px in x and
0.4 * 6k px in y. The moves are printed on stdout. If `dump` changes, this
file must change with it.

    python -m kibitz_tools.fake_run --kibitz ../target/release/kibitz.exe --plies 60 --runs 2 --out data/synth

writes data/synth/run0, run1, ... each holding a consecutive slice of the
game, in the `--run-dir` layout (JPEG q80 like the runtime, sidecars, game.pgn).
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from PIL import Image

W, H = 1280, 720
FRAMES_PER_PLY = 10
SHIFT_PER_PLY = 6.0


def default_corners(width: int = W, height: int = H) -> np.ndarray:
    """Port of `synth::default_params` corners [a8, h8, h1, a1], in f32 like Rust."""
    f = np.float32
    side = f(0.6) * f(height)
    cx, cy = f(width) / f(2), f(height) / f(2)
    half = side / f(2)
    top_half = side / f(2) * f(0.92)
    return np.array(
        [[cx - top_half, cy - half], [cx + top_half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
        np.float32,
    )


def dump_corners(ply: int) -> np.ndarray:
    """Corners of the frames showing the position after `ply` plies."""
    shift = np.float32(SHIFT_PER_PLY * ply)
    return default_corners() + np.array([shift, shift * np.float32(0.4)], np.float32)


def run_dump(kibitz: Path, plies: int, out: Path) -> list[str]:
    """Run `kibitz bench --dump` and return the scripted moves (UCI)."""
    res = subprocess.run(
        [str(kibitz), "bench", "--dump", str(out), "--frames", str(plies)],
        capture_output=True, text=True, check=True,
    )
    m = re.search(r"moves: (.*)", res.stdout)
    if not m:
        raise RuntimeError(f"unexpected bench output:\n{res.stdout}")
    return m.group(1).split()


def write_run(dump: Path, moves: list[str], first: int, last: int, out: Path, my_color: str, ts0: int):
    """Write plies first..last (inclusive, 1-based) of the dumped game as a run dir.
    Each ply gets one "commit" frame (the last, settled frame of its hold); the
    first ply also gets a "missed" frame with a deliberately stale fen, so
    loaders that use it as a label would be caught by the tests."""
    (out / "frames").mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    for uci in moves[: first - 1]:
        board.push_uci(uci)
    game = chess.pgn.Game()
    game.setup(board)
    node = game

    def save(ply: int, frame_idx: int, what: str, fen: str):
        ts = ts0 + frame_idx * 66
        gray = Image.open(dump / f"{frame_idx:05d}.png").convert("L")
        gray.convert("RGB").save(out / "frames" / f"{ts}-{what}.jpg", quality=80)
        meta = {
            "ts_ms": ts, "what": what, "width": W, "height": H,
            "corners": dump_corners(ply).tolist(), "fen": fen, "my_color": my_color,
        }
        (out / "frames" / f"{ts}-{what}.json").write_text(json.dumps(meta))

    for ply in range(first, last + 1):
        stale = board.fen()
        board.push_uci(moves[ply - 1])
        node = node.add_variation(board.move_stack[-1])
        if ply == first:
            save(ply, ply * FRAMES_PER_PLY + 2, "missed", stale)
        save(ply, ply * FRAMES_PER_PLY + FRAMES_PER_PLY - 1, "commit", board.fen())
    (out / "game.pgn").write_text(str(game) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kibitz", type=Path, required=True, help="path to the kibitz binary")
    ap.add_argument("--plies", type=int, default=60)
    ap.add_argument("--runs", type=int, default=2, help="split the game into this many run dirs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp)
        moves = run_dump(args.kibitz, args.plies, dump)
        n = len(moves)
        bounds = np.linspace(0, n, args.runs + 1).astype(int)
        for r in range(args.runs):
            first, last = bounds[r] + 1, bounds[r + 1]
            # Alternate colours so the Black-seat code paths run too (the warp is White's view either way).
            color = "white" if r % 2 == 0 else "black"
            write_run(dump, moves, first, last, args.out / f"run{r}", color, 1_700_000_000_000 + r * 10_000_000)
            print(f"run{r}: plies {first}..{last} ({color})")


if __name__ == "__main__":
    main()
