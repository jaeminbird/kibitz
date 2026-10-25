"""FEN labels, and loading a run dir (built here from the golden synthetic frames)."""

import json

import chess
import numpy as np
from PIL import Image

from kibitz_tools.board import square_rect, warp_board
from kibitz_tools.labels import CLASSES, NUM_CLASSES, changed_squares, fen_labels
from kibitz_tools.rundir import load_run, move_between

START = chess.STARTING_FEN


def test_start_labels():
    lab = fen_labels(START)
    assert NUM_CLASSES == 13
    assert CLASSES[lab[0]] == "R" and CLASSES[lab[4]] == "K"  # a1, e1
    assert CLASSES[lab[59]] == "q" and CLASSES[lab[63]] == "r"  # d8, h8
    assert all(CLASSES[lab[s]] == "P" for s in range(8, 16))
    assert np.all(lab[16:48] == 0)


def test_changed_squares_special_moves():
    # Castling: king and rook squares, from and to (e1, f1, g1, h1).
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    before = b.fen()
    b.push_uci("e1g1")
    assert changed_squares(before, b.fen()) == [4, 5, 6, 7]
    # En passant: from, to and the captured pawn (e5, d6, d5).
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    before = b.fen()
    b.push_uci("e5d6")
    assert changed_squares(before, b.fen()) == [35, 36, 43]


def test_move_between():
    b = chess.Board()
    b.push_uci("g1f3")
    assert move_between(START, b.fen()) == chess.Move.from_uci("g1f3")
    assert move_between(START, START) is None


def make_run(tmp_path, golden):
    """Two commit frames (start, after 1. e4) and one 'missed' frame with a wrong fen."""
    frames = tmp_path / "frames"
    frames.mkdir()
    after = chess.Board()
    after.push_uci("e2e4")
    for ts, what, img, fen in [
        (1000, "commit", golden["frame_a"], START),
        (1500, "missed", golden["frame_b"], START),  # wrong on purpose
        (2000, "commit", golden["frame_b"], after.fen()),
    ]:
        Image.fromarray(img).convert("RGB").save(frames / f"{ts}-{what}.jpg", quality=95)
        meta = dict(ts_ms=ts, what=what, width=img.shape[1], height=img.shape[0],
                    corners=golden["corners"], fen=fen, my_color="white")
        (frames / f"{ts}-{what}.json").write_text(json.dumps(meta))
    (tmp_path / "events.jsonl").write_text('{"ts_ms":5,"kind":"resumed"}\n')
    return tmp_path


def test_load_run(tmp_path, golden):
    run = load_run(make_run(tmp_path, golden))
    assert [f.what for f in run.frames] == ["commit", "missed", "commit"]
    assert len(run.commits()) == 2  # "missed" is never a label
    pairs = run.commit_pairs()
    assert len(pairs) == 1 and pairs[0][2] == chess.Move.from_uci("e2e4")
    assert run.events == [{"ts_ms": 5, "kind": "resumed"}]
    assert run.pgn is None


def test_warp_orientation_matches_labels(tmp_path, golden):
    # Synthetic pieces are discs at 240 (white) or 25 (black). After the warp,
    # the centre of each square must agree with the FEN label: this checks the
    # whole chain corners -> warp -> square index -> label.
    run = load_run(make_run(tmp_path, golden))
    for f in run.commits():
        warped = warp_board(f.gray(), f.corners, 32)
        lab = fen_labels(f.fen)
        yy, xx = np.mgrid[0:32, 0:32] + 0.5
        disc = (xx - 16) ** 2 + (yy - 16) ** 2 < (0.30 * 32) ** 2
        for s in range(64):
            x0, y0, x1, y1 = square_rect(s, 32)
            # Median over the central disc: the role ring covers at most ~30% of it
            # (the king's), so the median is the piece tone, or the square tone if empty.
            c = np.median(warped[y0:y1, x0:x1][disc])
            if 1 <= lab[s] <= 6:
                assert c > 215, (s, c)
            elif lab[s] >= 7:
                assert c < 70, (s, c)
            else:
                assert 75 < c < 215, (s, c)
