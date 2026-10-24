//! Per-square appearance features and their difference.
//!
//! Each square's inner region (15% trimmed on every side, so the board border
//! and the piece-free edge are ignored) is summarised by a 4×4 grid of cell
//! means plus one standard deviation. Pixels are normalised with the mean and
//! standard deviation of the whole warped board, which makes the features
//! insensitive to global exposure changes.

use crate::geometry::WarpedBoard;

/// Appearance summary of every square of a warped board.
pub struct SquareStats {
    /// Per square: 4×4 grid of cell means over the inner 70% of the square, in globally-normalised units.
    pub cells: [[f32; 16]; 64],
    /// Per square: std-dev of the inner region (normalised units).
    pub std: [f32; 64],
}

/// Summarise every square of `b`. Uses integer sums over the pixels; the only
/// allocation is the returned struct.
pub fn square_stats(b: &WarpedBoard) -> SquareStats {
    let n = b.size() as usize;
    let data = &b.img.data;

    // Global moments of the whole image, for normalisation.
    let (mut s1, mut s2) = (0u64, 0u64);
    for &p in data {
        let p = p as u64;
        s1 += p;
        s2 += p * p;
    }
    let count = data.len() as f64;
    let mean = s1 as f64 / count;
    let sigma = (s2 as f64 / count - mean * mean).max(0.0).sqrt().max(1.0);
    let inv_sigma = 1.0 / sigma;

    let sq = b.sq_px as usize;
    // Drop 15% on each side; keep at least one pixel so the arithmetic below is defined.
    let margin = (sq as f64 * 0.15).round() as usize;
    let inner = sq.saturating_sub(2 * margin).max(1);

    let mut out = SquareStats { cells: [[0.0; 16]; 64], std: [0.0; 64] };
    for s in 0..64u8 {
        let (x0, y0, _, _) = b.square_rect(s);
        let (x0, y0) = (x0 as usize + margin, y0 as usize + margin);

        let mut cell_sum = [0u64; 16];
        let mut cell_sq = [0u64; 16];
        let mut cell_n = [0u32; 16];
        for j in 0..inner {
            let cy = (4 * j) / inner;
            let row = &data[(y0 + j) * n + x0..(y0 + j) * n + x0 + inner];
            for (i, &p) in row.iter().enumerate() {
                let k = cy * 4 + (4 * i) / inner;
                let p = p as u64;
                cell_sum[k] += p;
                cell_sq[k] += p * p;
                cell_n[k] += 1;
            }
        }

        for k in 0..16 {
            // Cells are non-empty when inner >= 4; the max guards tiny squares.
            let cnt = cell_n[k].max(1) as f64;
            let m = cell_sum[k] as f64 / cnt;
            out.cells[s as usize][k] = ((m - mean) * inv_sigma) as f32;
        }

        let total_n = (inner * inner) as f64;
        let (ts, tq) = (cell_sum.iter().sum::<u64>() as f64, cell_sq.iter().sum::<u64>() as f64);
        let m = ts / total_n;
        let var = (tq / total_n - m * m).max(0.0);
        out.std[s as usize] = (var.sqrt() * inv_sigma) as f32;
    }
    out
}

/// Per-square change score between two summaries of the same board geometry:
/// `mean_k |cells_ref − cells_cur| + |std_ref − std_cur|`.
pub fn square_diff(reference: &SquareStats, current: &SquareStats) -> [f32; 64] {
    let mut out = [0f32; 64];
    for (s, o) in out.iter_mut().enumerate() {
        let cell_diff: f32 =
            reference.cells[s].iter().zip(&current.cells[s]).map(|(a, b)| (a - b).abs()).sum();
        *o = cell_diff / 16.0 + (reference.std[s] - current.std[s]).abs();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{warp_board, GrayImage, WarpedBoard};
    use cg_core::Corners;

    /// Board with a dark/light checker texture, a bright piece-like disc on `bright`
    /// squares, and a gradient on the outside.
    fn test_board(sq_px: u32, bright: &[u8]) -> WarpedBoard {
        let n = sq_px * 8;
        let mut img = GrayImage::new(n, n);
        for y in 0..n {
            for x in 0..n {
                let (col, row) = (x / sq_px, y / sq_px);
                let s = crate::geometry::grid_to_square(row, col);
                let light = (s / 8 + s % 8) % 2 == 1;
                let mut v = if light { 190.0 } else { 95.0 };
                if bright.contains(&s) {
                    let dx = (x % sq_px) as f32 / sq_px as f32 - 0.5;
                    let dy = (y % sq_px) as f32 / sq_px as f32 - 0.5;
                    if dx * dx + dy * dy < 0.13 {
                        v = 245.0;
                    }
                }
                img.data[(y * n + x) as usize] = v as u8;
            }
        }
        WarpedBoard { sq_px, img }
    }

    #[test]
    fn identical_images_have_zero_diff() {
        let b = test_board(32, &[12, 28]);
        let s = square_stats(&b);
        assert!(square_diff(&s, &s).iter().all(|&d| d.abs() < 1e-6));
    }

    #[test]
    fn brightness_offset_is_invisible() {
        let b = test_board(32, &[12]);
        let mut b2 = b.clone();
        // Test values top out at 245, so +8 never clips.
        for p in b2.img.data.iter_mut() {
            *p += 8;
        }
        let d = square_diff(&square_stats(&b), &square_stats(&b2));
        assert!(d.iter().all(|&x| x < 1e-3), "{d:?}");
    }

    #[test]
    fn moved_piece_is_detected_locally() {
        let before = test_board(32, &[12]); // piece on e2
        let after = test_board(32, &[28]); // piece on e4
        let d = square_diff(&square_stats(&before), &square_stats(&after));
        for s in [12usize, 28] {
            assert!(d[s] > 0.3, "square {s} diff {}", d[s]);
        }
        for (s, &v) in d.iter().enumerate() {
            if s != 12 && s != 28 {
                assert!(v < 0.05, "square {s} diff {v}");
            }
        }
    }

    #[test]
    fn warped_gradient_stats_are_finite() {
        let w = 200u32;
        let gray: Vec<u8> = (0..w * w).map(|i| ((i * 7) % 256) as u8).collect();
        let c = Corners([[10.0, 10.0], [190.0, 12.0], [188.0, 190.0], [12.0, 188.0]]);
        let wb = warp_board(&gray, w, w, &c, 16).unwrap();
        let s = square_stats(&wb);
        assert!(s.cells.iter().flatten().all(|v| v.is_finite()));
        assert!(s.std.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn stats_timing_release() {
        let b = test_board(32, &[0, 9, 18, 27, 36, 45, 54, 63]);
        let b2 = test_board(32, &[1, 10, 19, 28, 37, 46, 55]);
        let iters = 500;
        let t = std::time::Instant::now();
        let mut sink = 0f32;
        for _ in 0..iters {
            sink += square_diff(&square_stats(&b), &square_stats(&b2))[12];
        }
        let per = t.elapsed().as_secs_f64() / (2.0 * iters as f64) * 1e3;
        std::hint::black_box(sink);
        println!("square_stats (256x256 board): {per:.4} ms per call");
    }
}
