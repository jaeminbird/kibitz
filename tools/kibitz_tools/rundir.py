"""Load a run directory written by `kibitz run --run-dir DIR`.

Layout (see `save_frame` / `save_pgn` in crates/cg-app/src/orchestrator.rs):

    DIR/game.pgn
    DIR/frames/<ts_ms>-<what>.jpg
    DIR/frames/<ts_ms>-<what>.json   {ts_ms, what, width, height, corners, fen, my_color}
    DIR/events.jsonl                 optional, from `--log` (cg-core/src/event.rs)

`what` is "commit" (board just after a committed move; `fen` is that position,
so it is a reliable label), or "lost" / "missed" (`fen` may not match the
physical board: never use those as labels).
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from PIL import Image

from .board import to_gray


@dataclass
class FrameRecord:
    ts_ms: int
    what: str
    width: int
    height: int
    # [[x, y] x 4] in frame pixels, board order [a8, h8, h1, a1]; None if the board wasn't located.
    corners: list | None
    fen: str
    my_color: str
    jpg: Path

    def gray(self) -> np.ndarray:
        """The frame as Luma8 (H, W), converted like the runtime does."""
        return to_gray(self.rgb())

    def rgb(self) -> np.ndarray:
        return np.asarray(Image.open(self.jpg).convert("RGB"))

    @property
    def is_label(self) -> bool:
        """True when `fen` is a trustworthy label for what is on the board."""
        return self.what == "commit" and self.corners is not None


@dataclass
class Run:
    path: Path
    frames: list[FrameRecord]
    pgn: chess.pgn.Game | None
    events: list[dict]

    @property
    def name(self) -> str:
        return self.path.name

    def commits(self) -> list[FrameRecord]:
        """Labelled frames ("commit" with corners), in time order."""
        return [f for f in self.frames if f.is_label]

    def commit_pairs(self) -> list[tuple[FrameRecord, FrameRecord, chess.Move]]:
        """Consecutive commit frames exactly one legal move apart, with that move.
        Pairs across an undo, a new game or a missed commit frame are skipped."""
        out = []
        cs = self.commits()
        for a, b in zip(cs, cs[1:]):
            mv = move_between(a.fen, b.fen)
            if mv is not None:
                out.append((a, b, mv))
        return out


def move_between(before_fen: str, after_fen: str) -> chess.Move | None:
    """The legal move that turns `before_fen` into `after_fen` (piece placement
    only), or None if no single move does."""
    board = chess.Board(before_fen)
    target = chess.Board(after_fen).board_fen()
    for mv in board.legal_moves:
        board.push(mv)
        hit = board.board_fen() == target
        board.pop()
        if hit:
            return mv
    return None


def load_run(path, events: str | Path | None = None) -> Run:
    """Load a run directory. `events` is an events.jsonl path; by default
    `DIR/events.jsonl` is used if it exists."""
    path = Path(path)
    frames = []
    for side in sorted((path / "frames").glob("*.json")):
        meta = json.loads(side.read_text())
        jpg = side.with_suffix(".jpg")
        if not jpg.exists():
            continue  # sidecar written but the JPEG write failed or is still pending
        frames.append(
            FrameRecord(
                ts_ms=int(meta["ts_ms"]),
                what=meta["what"],
                width=int(meta["width"]),
                height=int(meta["height"]),
                corners=meta.get("corners"),
                fen=meta["fen"],
                my_color=meta.get("my_color", "white"),
                jpg=jpg,
            )
        )
    frames.sort(key=lambda f: (f.ts_ms, f.what))

    pgn = None
    if (path / "game.pgn").exists():
        pgn = chess.pgn.read_game(io.StringIO((path / "game.pgn").read_text()))

    ev_path = Path(events) if events else path / "events.jsonl"
    evs = []
    if ev_path.exists():
        for line in ev_path.read_text().splitlines():
            if line.strip():
                evs.append(json.loads(line))
    return Run(path=path, frames=frames, pgn=pgn, events=evs)
