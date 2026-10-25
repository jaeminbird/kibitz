"""FEN -> per-square class labels, and the squares a move changes.

Classes (13): 0 = empty, 1..6 = white pawn, knight, bishop, rook, queen, king,
7..12 = the same for black. This order is part of the classifier contract
(classifier/MODEL_CONTRACT.md).
"""

import chess
import numpy as np

CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
NUM_CLASSES = len(CLASSES)


def piece_class(piece: chess.Piece | None) -> int:
    """Class index of a python-chess piece (None = empty)."""
    if piece is None:
        return 0
    # python-chess piece_type: PAWN = 1 ... KING = 6.
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


def fen_labels(fen: str) -> np.ndarray:
    """64 class labels (uint8) in shakmaty / python-chess square order (a1 = 0)."""
    board = chess.Board(fen)
    return np.array([piece_class(board.piece_at(s)) for s in range(64)], np.uint8)


def changed_squares(before_fen: str, after_fen: str) -> list[int]:
    """Squares whose occupant differs between two positions. For one legal move
    this is the same set as `changed_squares` in infer.rs (from, to, castling
    rook squares, en-passant victim)."""
    a, b = fen_labels(before_fen), fen_labels(after_fen)
    return [int(s) for s in np.nonzero(a != b)[0]]
