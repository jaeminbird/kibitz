"""The numpy ports in kibitz_tools/board.py match the Rust code.

Golden data comes from crates/cg-vision/tests/golden.rs: two synthetic frames
(start position; after 1. e4 with +8 brightness and other noise), a perspective
quad with one corner outside the frame, at sq_px 32 (runtime) and 30 (margin
rounding of 4.5).
"""

import numpy as np
import pytest

from kibitz_tools.board import square_diff, square_stats, warp_board


def cases(golden):
    return [(c["sq_px"], c) for c in golden["cases"]]


@pytest.mark.parametrize("which", ["a", "b"])
def test_warp_matches_rust(golden, which):
    for sq, case in cases(golden):
        mine = warp_board(golden[f"frame_{which}"], golden["corners"], sq).astype(int)
        rust = case[f"warped_{which}"].astype(int)
        assert mine.shape == rust.shape
        err = np.abs(mine - rust)
        # Same f64 arithmetic in the same order, so normally bit-exact; ±1 allowed for rounding ties.
        assert err.max() <= 1, f"sq_px {sq}: max error {err.max()}"
        print(f"warp {which} sq_px={sq}: {np.mean(err == 0):.4%} pixels exact, max error {err.max()}")


def test_out_of_frame_is_zero(golden):
    # The h1 corner is past the right edge of the frame, so the bottom-right of the warp is 0 in both.
    sq, case = cases(golden)[0]
    assert case["warped_a"][-1, -1] == 0
    assert warp_board(golden["frame_a"], golden["corners"], sq)[-1, -1] == 0


@pytest.mark.parametrize("which", ["a", "b"])
def test_stats_match_rust(golden, which):
    # On Rust's own warped board, so this checks square_stats alone.
    for sq, case in cases(golden):
        cells, std = square_stats(case[f"warped_{which}"], sq)
        ref = case[f"stats_{which}"]
        np.testing.assert_allclose(cells, np.array(ref["cells"], np.float32), atol=1e-3, rtol=0)
        np.testing.assert_allclose(std, np.array(ref["std"], np.float32), atol=1e-3, rtol=0)
        print(f"stats {which} sq_px={sq}: max |cells| err {np.abs(cells - np.array(ref['cells'])).max():.2e}")


def test_diff_matches_rust_end_to_end(golden):
    # Frame -> warp -> stats -> diff entirely in Python, against Rust's diff.
    for sq, case in cases(golden):
        a = square_stats(warp_board(golden["frame_a"], golden["corners"], sq), sq)
        b = square_stats(warp_board(golden["frame_b"], golden["corners"], sq), sq)
        d = square_diff(a, b)
        np.testing.assert_allclose(d, np.array(case["diff"], np.float32), atol=1e-3, rtol=0)
        # And the diff means something: e2 (12) and e4 (28) changed the most.
        assert set(np.argsort(d)[-2:]) == {12, 28}
