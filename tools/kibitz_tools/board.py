"""numpy port of the board geometry and square features.

Mirrors `crates/cg-vision/src/geometry.rs` and `features.rs`; see those files
for the reasoning. `tests/test_parity.py` checks this module against golden
output from the Rust code (`cargo test -p cg-vision --test golden -- --ignored`).

Warped board convention (same as Rust): a square `N x N` grayscale image,
`N = 8 * sq_px`, always from White's point of view: row 0 is rank 8, column 0
is file a. Square indices use shakmaty's numbering (a1 = 0, b1 = 1, ..., h8 = 63),
which is also python-chess's numbering.
"""

import numpy as np

# Runtime square size, `SQ_PX` in crates/cg-app/src/vision.rs.
SQ_PX = 32


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB8 (H, W, 3) -> Luma8 (H, W), the same BT.601 integer approximation
    as `luma` in crates/cg-core/src/frame.rs."""
    rgb = rgb.astype(np.uint32)
    return ((rgb[..., 0] * 77 + rgb[..., 1] * 150 + rgb[..., 2] * 29) >> 8).astype(np.uint8)


def homography_from_points(src, dst) -> np.ndarray | None:
    """Solve for H (3x3, H[2,2] = 1) such that H * src[i] ~ dst[i] for four
    point pairs. None for degenerate input. Same elimination as the Rust code,
    so results agree to the last bits."""
    a = np.zeros((8, 9))
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u]
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v]
    # Gaussian elimination with partial pivoting on the augmented 8x9 matrix.
    for col in range(8):
        pivot = col + int(np.argmax(np.abs(a[col:, col])))
        if abs(a[pivot, col]) < 1e-12:
            return None
        a[[col, pivot]] = a[[pivot, col]]
        for row in range(8):
            if row != col:
                f = a[row, col] / a[col, col]
                if f != 0.0:
                    a[row, col:] -= f * a[col, col:]
    h = np.ones(9)
    h[:8] = a[:, 8] / np.diag(a[:, :8])
    return h.reshape(3, 3)


def board_to_image(corners, size: int) -> np.ndarray | None:
    """Homography mapping warped-board pixel coords -> image pixel coords.
    `corners` are [a8, h8, h1, a1] in image pixels (the sidecar `corners`).
    They are f32 in Rust, so round-trip through float32 to get the same values."""
    c = np.asarray(corners, dtype=np.float32).astype(np.float64)
    n = float(size)
    return homography_from_points([[0, 0], [n, 0], [n, n], [0, n]], c)


def warp_board(gray: np.ndarray, corners, sq_px: int = SQ_PX, pad_px: int = 0) -> np.ndarray | None:
    """Rectify the board region of a grayscale frame (H, W) uint8.

    Bilinear sampling; pixels outside the frame are 0. With `pad_px = 0` this
    is exactly Rust's `warp_board` (an `8*sq_px` square). `pad_px > 0` also
    samples a border of that many pixels around the board with the same
    homography (output `8*sq_px + 2*pad_px`), so edge squares get context for
    classifier crops. Board pixel (u, v) is then at output (u + pad, v + pad).
    """
    size = sq_px * 8
    h = board_to_image(corners, size)
    if h is None:
        return None
    out_n = size + 2 * pad_px
    # Pixel centres in warped coordinates, relative to the board's top-left corner.
    uu, vv = np.meshgrid(np.arange(out_n) - pad_px + 0.5, np.arange(out_n) - pad_px + 0.5)
    w = h[2, 0] * uu + h[2, 1] * vv + h[2, 2]
    x = (h[0, 0] * uu + h[0, 1] * vv + h[0, 2]) / w - 0.5
    y = (h[1, 0] * uu + h[1, 1] * vv + h[1, 2]) / w - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)

    hgt, wid = gray.shape
    img = gray.astype(np.float64)

    def px(xx, yy):
        inside = (xx >= 0) & (yy >= 0) & (xx < wid) & (yy < hgt)
        return np.where(inside, img[np.clip(yy, 0, hgt - 1), np.clip(xx, 0, wid - 1)], 0.0)

    val = (
        px(x0, y0) * (1 - fx) * (1 - fy)
        + px(x0 + 1, y0) * fx * (1 - fy)
        + px(x0, y0 + 1) * (1 - fx) * fy
        + px(x0 + 1, y0 + 1) * fx * fy
    )
    # Rust's f64::round is half away from zero; values are >= 0 so floor(v + 0.5) matches.
    return np.clip(np.floor(val + 0.5), 0, 255).astype(np.uint8)


def square_to_grid(square: int) -> tuple[int, int]:
    """shakmaty square index (a1 = 0) -> warped grid (row, col) with row 0 = rank 8."""
    return 7 - square // 8, square % 8


def grid_to_square(row: int, col: int) -> int:
    """Warped grid (row, col) -> shakmaty square index."""
    return (7 - row) * 8 + col


def square_rect(square: int, sq_px: int = SQ_PX) -> tuple[int, int, int, int]:
    """Pixel rectangle (x0, y0, x1, y1) of a square in the warped board, exclusive end."""
    row, col = square_to_grid(square)
    return col * sq_px, row * sq_px, col * sq_px + sq_px, row * sq_px + sq_px


def _round_half_away(x: float) -> int:
    # Python's round() is banker's rounding; Rust's is half away from zero.
    return int(np.floor(x + 0.5))


def square_stats(warped: np.ndarray, sq_px: int = SQ_PX) -> tuple[np.ndarray, np.ndarray]:
    """Port of `square_stats` (features.rs). Returns (cells (64, 16) f32, std (64,) f32).

    Each square's inner region (15% trimmed on every side) is summarised by a
    4x4 grid of cell means plus one standard deviation, in units normalised by
    the whole warped board's mean and standard deviation.
    """
    data = warped.astype(np.int64)
    count = data.size
    mean = data.sum() / count
    sigma = max(np.sqrt(max((data * data).sum() / count - mean * mean, 0.0)), 1.0)
    inv_sigma = 1.0 / sigma

    margin = _round_half_away(sq_px * 0.15)
    inner = max(sq_px - 2 * margin, 1)
    # Cell index of every pixel of the inner region (row-major 4x4 grid).
    j = (4 * np.arange(inner)) // inner
    cell_of = (j[:, None] * 4 + j[None, :]).ravel()

    cells = np.zeros((64, 16), np.float32)
    std = np.zeros(64, np.float32)
    for s in range(64):
        x0, y0, _, _ = square_rect(s, sq_px)
        region = data[y0 + margin : y0 + margin + inner, x0 + margin : x0 + margin + inner].ravel()
        cell_sum = np.bincount(cell_of, weights=region, minlength=16)
        cell_n = np.maximum(np.bincount(cell_of, minlength=16), 1)
        cells[s] = ((cell_sum / cell_n - mean) * inv_sigma).astype(np.float32)
        m = region.sum() / region.size
        var = max((region * region).sum() / region.size - m * m, 0.0)
        std[s] = np.float32(np.sqrt(var) * inv_sigma)
    return cells, std


def square_diff(reference: tuple[np.ndarray, np.ndarray], current: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Per-square change score (64,) f32: mean_k |cells_ref - cells_cur| + |std_ref - std_cur|."""
    (rc, rs), (cc, cs) = reference, current
    return (np.abs(rc - cc).sum(axis=1) / np.float32(16) + np.abs(rs - cs)).astype(np.float32)
