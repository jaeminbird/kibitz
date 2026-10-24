//! Planar homography and board warping.
//!
//! Warped board convention: a square `N×N` grayscale image, `N = 8 * sq_px`,
//! always from White's point of view: row 0 is rank 8, column 0 is file a.
//! Square indices everywhere in this crate use shakmaty's numbering
//! (a1 = 0, b1 = 1, …, h8 = 63).

use cg_core::Corners;

/// 3×3 projective transform, row-major, `h[8]` normalised to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    /// Solve for H such that `H * src[i] ~ dst[i]` for four point pairs.
    /// Returns None for degenerate input (three collinear points etc.).
    #[allow(clippy::needless_range_loop)] // index-based elimination reads clearer
    pub fn from_points(src: [[f64; 2]; 4], dst: [[f64; 2]; 4]) -> Option<Homography> {
        // Unknowns h0..h7 (h8 = 1). Two equations per correspondence.
        let mut a = [[0f64; 9]; 8];
        for i in 0..4 {
            let [x, y] = src[i];
            let [u, v] = dst[i];
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        // Gaussian elimination with partial pivoting on the augmented 8×9 matrix.
        for col in 0..8 {
            let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
            if a[pivot][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, pivot);
            for row in 0..8 {
                if row != col {
                    let f = a[row][col] / a[col][col];
                    if f != 0.0 {
                        for k in col..9 {
                            a[row][k] -= f * a[col][k];
                        }
                    }
                }
            }
        }
        let mut h = [0f64; 9];
        for i in 0..8 {
            h[i] = a[i][8] / a[i][i];
        }
        h[8] = 1.0;
        Some(Homography(h))
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
    }
}

/// Grayscale image owned buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: u32, height: u32) -> GrayImage {
        GrayImage { width, height, data: vec![0; (width * height) as usize] }
    }
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.data[(y * self.width + x) as usize]
    }
}

/// A board rectified to `8*sq_px` square pixels (see module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct WarpedBoard {
    pub sq_px: u32,
    pub img: GrayImage,
}

impl WarpedBoard {
    pub fn size(&self) -> u32 {
        self.sq_px * 8
    }

    /// Pixel rectangle (x0, y0, x1, y1) of a square, exclusive end.
    pub fn square_rect(&self, square: u8) -> (u32, u32, u32, u32) {
        let (row, col) = square_to_grid(square);
        let s = self.sq_px;
        (col * s, row * s, col * s + s, row * s + s)
    }
}

/// shakmaty square index (a1 = 0) → warped grid (row, col) with row 0 = rank 8.
#[inline]
pub fn square_to_grid(square: u8) -> (u32, u32) {
    let file = (square % 8) as u32;
    let rank = (square / 8) as u32;
    (7 - rank, file)
}

/// Warped grid (row, col) → shakmaty square index.
#[inline]
pub fn grid_to_square(row: u32, col: u32) -> u8 {
    ((7 - row) * 8 + col) as u8
}

/// Homography mapping warped-board pixel coords → image pixel coords.
pub fn board_to_image(corners: &Corners, size: u32) -> Option<Homography> {
    let n = size as f64;
    let dst = corners.0.map(|[x, y]| [x as f64, y as f64]);
    Homography::from_points([[0.0, 0.0], [n, 0.0], [n, n], [0.0, n]], dst)
}

/// Rectify the board region of a grayscale frame. Bilinear sampling;
/// pixels that fall outside the frame are 0.
pub fn warp_board(gray: &[u8], width: u32, height: u32, corners: &Corners, sq_px: u32) -> Option<WarpedBoard> {
    let size = sq_px * 8;
    let h = board_to_image(corners, size)?;
    let mut out = GrayImage::new(size, size);
    let (w, hgt) = (width as i64, height as i64);
    for v in 0..size {
        for u in 0..size {
            let (x, y) = h.apply(u as f64 + 0.5, v as f64 + 0.5);
            let (x, y) = (x - 0.5, y - 0.5);
            let x0 = x.floor();
            let y0 = y.floor();
            let (fx, fy) = (x - x0, y - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let px = |xx: i64, yy: i64| -> f64 {
                if xx < 0 || yy < 0 || xx >= w || yy >= hgt {
                    0.0
                } else {
                    gray[(yy * w + xx) as usize] as f64
                }
            };
            let val = px(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + px(x0 + 1, y0) * fx * (1.0 - fy)
                + px(x0, y0 + 1) * (1.0 - fx) * fy
                + px(x0 + 1, y0 + 1) * fx * fy;
            out.data[(v * size + u) as usize] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    Some(WarpedBoard { sq_px, img: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_corners() {
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let dst = [[10.0, 20.0], [200.0, 30.0], [180.0, 220.0], [5.0, 190.0]];
        let h = Homography::from_points(src, dst).unwrap();
        for i in 0..4 {
            let (x, y) = h.apply(src[i][0], src[i][1]);
            assert!((x - dst[i][0]).abs() < 1e-6 && (y - dst[i][1]).abs() < 1e-6);
        }
    }

    #[test]
    fn degenerate_is_none() {
        let src = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        assert!(Homography::from_points(src, src).is_none());
    }

    #[test]
    fn square_mapping() {
        assert_eq!(square_to_grid(0), (7, 0)); // a1 bottom-left
        assert_eq!(square_to_grid(63), (0, 7)); // h8 top-right
        for s in 0..64u8 {
            let (r, c) = square_to_grid(s);
            assert_eq!(grid_to_square(r, c), s);
        }
    }

    #[test]
    fn identity_warp_copies() {
        // 64x64 image with a gradient; corners = full image → warp at sq_px=8 is a copy.
        let w = 64u32;
        let gray: Vec<u8> = (0..w * w).map(|i| ((i % w) * 4) as u8).collect();
        let c = Corners([[0.0, 0.0], [64.0, 0.0], [64.0, 64.0], [0.0, 64.0]]);
        let wb = warp_board(&gray, w, w, &c, 8).unwrap();
        assert_eq!(wb.img.data, gray);
    }
}
