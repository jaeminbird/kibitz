"""Classifier crop geometry (kibitz_tools/crops.py, classifier/MODEL_CONTRACT.md)."""

import numpy as np

from kibitz_tools.board import SQ_PX, square_rect, warp_board
from kibitz_tools.crops import CROP_H, CROP_W, PAD, SIDE, TOP, board_moments, oriented_warp, square_crops


def test_crop_shape_and_square_position(golden):
    frame, corners = golden["frame_a"], golden["corners"]
    plain = warp_board(frame, corners, SQ_PX)
    for color in ["white", "black"]:
        crops = square_crops(oriented_warp(frame, corners, color), color)
        assert crops.shape == (64, CROP_H, CROP_W) == (64, 56, 48)
        for s in [0, 12, 36, 63]:
            x0, y0, x1, y1 = square_rect(s, SQ_PX)
            sq = plain[y0:y1, x0:x1]
            inner = crops[s, TOP : TOP + SQ_PX, SIDE : SIDE + SQ_PX]
            # White: the square as warped. Black: rotated 180 degrees (the player's view).
            np.testing.assert_array_equal(inner, sq if color == "white" else sq[::-1, ::-1])


def test_context_above_is_the_far_side(golden):
    frame, corners = golden["frame_a"], golden["corners"]
    plain = warp_board(frame, corners, SQ_PX, pad_px=PAD)
    # White player, e2 (12): the context above the square is e3, i.e. toward rank 8.
    crop = square_crops(oriented_warp(frame, corners, "white"), "white")[12]
    x0, y0, _, _ = square_rect(12, SQ_PX)
    above = plain[PAD + y0 - TOP : PAD + y0, PAD + x0 - SIDE : PAD + x0 + SQ_PX + SIDE]
    np.testing.assert_array_equal(crop[:TOP], above)
    # Black player, e7 (52): above is e6, i.e. toward rank 1, which is below e7 in the white-view warp.
    crop = square_crops(oriented_warp(frame, corners, "black"), "black")[52]
    x0, y0, _, y1 = square_rect(52, SQ_PX)
    below_in_white_view = plain[PAD + y1 : PAD + y1 + TOP, PAD + x0 - SIDE : PAD + x0 + SQ_PX + SIDE]
    np.testing.assert_array_equal(crop[:TOP], below_in_white_view[::-1, ::-1])


def test_board_moments_ignore_padding(golden):
    frame, corners = golden["frame_a"], golden["corners"]
    plain = warp_board(frame, corners, SQ_PX).astype(np.float64)
    mean, std = board_moments(oriented_warp(frame, corners, "black"))
    assert abs(mean - plain.mean()) < 1e-9 and abs(std - plain.std()) < 1e-9
