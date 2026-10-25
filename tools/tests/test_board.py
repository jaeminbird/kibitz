"""Unit tests for kibitz_tools.board, mirroring the Rust tests in geometry.rs / features.rs."""

import numpy as np

from kibitz_tools.board import (
    grid_to_square,
    homography_from_points,
    square_diff,
    square_rect,
    square_stats,
    square_to_grid,
    to_gray,
    warp_board,
)


def test_homography_maps_corners():
    src = [[0, 0], [100, 0], [100, 100], [0, 100]]
    dst = [[10, 20], [200, 30], [180, 220], [5, 190]]
    h = homography_from_points(src, dst)
    for (x, y), (u, v) in zip(src, dst):
        p = h @ [x, y, 1]
        assert abs(p[0] / p[2] - u) < 1e-6 and abs(p[1] / p[2] - v) < 1e-6


def test_degenerate_is_none():
    src = [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert homography_from_points(src, src) is None


def test_square_mapping():
    assert square_to_grid(0) == (7, 0)  # a1 bottom-left
    assert square_to_grid(63) == (0, 7)  # h8 top-right
    assert all(grid_to_square(*square_to_grid(s)) == s for s in range(64))
    assert square_rect(0, 32) == (0, 224, 32, 256)  # a1
    assert square_rect(12, 32) == (128, 192, 160, 224)  # e2


def test_identity_warp_copies():
    # 64x64 gradient; corners = full image -> warp at sq_px = 8 is a copy.
    w = 64
    gray = ((np.arange(w * w) % w) * 4).astype(np.uint8).reshape(w, w)
    out = warp_board(gray, [[0, 0], [64, 0], [64, 64], [0, 64]], 8)
    np.testing.assert_array_equal(out, gray)


def test_padded_warp_contains_plain_warp():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (200, 300), dtype=np.uint8)
    corners = [[60, 20], [250, 30], [270, 190], [40, 180]]
    plain = warp_board(gray, corners, 16)
    padded = warp_board(gray, corners, 16, pad_px=8)
    assert padded.shape == (144, 144)
    np.testing.assert_array_equal(padded[8:-8, 8:-8], plain)


def test_to_gray_matches_runtime_luma():
    rgb = np.array([[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    # (77 r + 150 g + 29 b) >> 8
    assert to_gray(rgb).tolist() == [[255, 76, 149, 28]]


def checker_board(sq_px, bright):
    """Checker texture with a bright disc on `bright` squares (features.rs test_board)."""
    n = sq_px * 8
    y, x = np.mgrid[0:n, 0:n]
    sq = (7 - y // sq_px) * 8 + x // sq_px
    img = np.where((sq // 8 + sq % 8) % 2 == 1, 190, 95).astype(np.uint8)
    dx = (x % sq_px) / sq_px - 0.5
    dy = (y % sq_px) / sq_px - 0.5
    img[np.isin(sq, bright) & (dx * dx + dy * dy < 0.13)] = 245
    return img


def test_identical_images_have_zero_diff():
    s = square_stats(checker_board(32, [12, 28]))
    assert np.all(np.abs(square_diff(s, s)) < 1e-6)


def test_brightness_offset_is_invisible():
    b = checker_board(32, [12])
    d = square_diff(square_stats(b), square_stats(b + 8))
    assert np.all(d < 1e-3)


def test_moved_piece_is_detected_locally():
    d = square_diff(square_stats(checker_board(32, [12])), square_stats(checker_board(32, [28])))
    assert d[12] > 0.3 and d[28] > 0.3
    assert np.all(np.delete(d, [12, 28]) < 0.05)
