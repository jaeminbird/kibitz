//! Synthetic board renderer for tests and benchmarks.
//!
//! Draws a checkerboard with pieces (discs with a role-dependent inner ring),
//! a 0.4-square frame, a smooth background, and additive noise. Rendering is
//! by inverse mapping: each image pixel is mapped into board coordinates.

use cg_core::Corners;
use shakmaty::{Board, Color, Role};

use crate::geometry::{grid_to_square, Homography};

/// Rendering parameters.
pub struct SynthParams {
    pub width: u32,
    pub height: u32,
    /// Board corners in board order [a8,h8,h1,a1], image pixels.
    pub corners: Corners,
    /// Gaussian-ish pixel noise std-dev in gray levels.
    pub noise: f32,
    /// Added to every pixel (simulates exposure change).
    pub brightness: f32,
    pub seed: u64,
}

/// Deterministic xorshift64* generator. Crate-internal, used by tests.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        // Mix the seed so nearby seeds give unrelated streams; never zero.
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        if s == 0 {
            s = 1;
        }
        Rng(s)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform index in 0..n.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Inner-ring radius (in square units) for each role, pawn to king.
fn ring_radius(role: Role) -> f32 {
    match role {
        Role::Pawn => 0.10,
        Role::Knight => 0.14,
        Role::Bishop => 0.18,
        Role::Rook => 0.22,
        Role::Queen => 0.26,
        Role::King => 0.30,
    }
}

/// Render `board` as a Luma8 image of `p.width × p.height`.
pub fn render(board: &Board, p: &SynthParams) -> Vec<u8> {
    let (w, h) = (p.width as usize, p.height as usize);
    // Map image pixels to board coordinates: the playing area is [0,8)² in these units.
    let to_board = Homography::from_points(
        p.corners.0.map(|[x, y]| [x as f64, y as f64]),
        [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]],
    )
    .expect("degenerate board corners");

    let mut pieces = [None; 64];
    for (sq, piece) in board.iter() {
        pieces[sq as usize] = Some((piece.color, piece.role));
    }

    // Background texture: three low-frequency sines with seed-derived phases.
    let mut rng = Rng::new(p.seed);
    let scale = 2.0 * std::f32::consts::PI / (w.max(h) as f32);
    // (frequency, phase) pairs for the three waves.
    let mut waves = [(0f32, 0f32); 3];
    for wv in waves.iter_mut() {
        let k = 1.0 + rng.below(4) as f32;
        *wv = (k * scale, rng.unit() * std::f32::consts::TAU);
    }

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (u, v) = to_board.apply(x as f64 + 0.5, y as f64 + 0.5);
            let (u, v) = (u as f32, v as f32);
            let base = if (0.0..8.0).contains(&u) && (0.0..8.0).contains(&v) {
                let (col, row) = (u.floor() as u32, v.floor() as u32);
                let s = grid_to_square(row, col) as usize;
                let (dx, dy) = (u - col as f32 - 0.5, v - row as f32 - 0.5);
                let light = (s / 8 + s % 8) % 2 == 1;
                let mut tone = if light { 190.0 } else { 95.0 };
                if let Some((color, role)) = pieces[s] {
                    let r2 = dx * dx + dy * dy;
                    let piece_tone = if color == Color::White { 240.0 } else { 25.0 };
                    if r2 <= 0.36 * 0.36 {
                        tone = piece_tone;
                        let d = r2.sqrt();
                        let r = ring_radius(role);
                        if d <= r && d >= r - 0.05 {
                            tone = if color == Color::White { 25.0 } else { 240.0 };
                        }
                    }
                }
                tone
            } else if (-0.4..8.4).contains(&u) && (-0.4..8.4).contains(&v) {
                120.0
            } else {
                let (x, y) = (x as f32, y as f32);
                let [(f1, p1), (f2, p2), (f3, p3)] = waves;
                let s = (x * f1 + p1).sin() + (y * f2 + p2).sin() + ((x + y) * f3 + p3).sin();
                120.0 + 25.0 * s / 3.0
            };

            // Noise: sum of four uniforms, centred and scaled to std-dev `noise`.
            let n4 = rng.unit() + rng.unit() + rng.unit() + rng.unit();
            let noise = (n4 - 2.0) * 3f32.sqrt() * p.noise;
            out[y * w + x] = (base + p.brightness + noise).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Parameters for a `width × height` image: board about 60% of the height,
/// centred, with a slight perspective (top edge narrower than the bottom).
pub fn default_params(width: u32, height: u32) -> SynthParams {
    let side = 0.6 * height as f32;
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let (half, top_half) = (side / 2.0, side / 2.0 * 0.92);
    let corners = Corners([
        [cx - top_half, cy - half], // a8
        [cx + top_half, cy - half], // h8
        [cx + half, cy + half],     // h1
        [cx - half, cy + half],     // a1
    ]);
    SynthParams { width, height, corners, noise: 3.0, brightness: 0.0, seed: 1 }
}

/// Paint a skin-tone (gray 150) ellipse centred at (`cx`, `cy`) with horizontal
/// radius `r` and vertical radius `0.7 r`.
pub fn render_hand(img: &mut [u8], width: u32, height: u32, cx: f32, cy: f32, r: f32) {
    assert_eq!(img.len(), (width * height) as usize);
    let ry = 0.7 * r;
    let x0 = (cx - r).floor().max(0.0) as u32;
    let x1 = ((cx + r).ceil().max(0.0) as u32).min(width);
    let y0 = (cy - ry).floor().max(0.0) as u32;
    let y1 = ((cy + ry).ceil().max(0.0) as u32).min(height);
    for y in y0..y1 {
        for x in x0..x1 {
            let dx = (x as f32 + 0.5 - cx) / r;
            let dy = (y as f32 + 0.5 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                img[(y * width + x) as usize] = 150;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{board_to_image, square_to_grid};
    use shakmaty::Position;

    fn startpos_board() -> Board {
        shakmaty::Chess::default().board().clone()
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let u = a.unit();
            assert_eq!(u, b.unit());
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn render_has_expected_size_and_tones() {
        let p = SynthParams { noise: 0.0, ..default_params(640, 480) };
        let board = startpos_board();
        let img = render(&board, &p);
        assert_eq!(img.len(), 640 * 480);

        // Sample square centres: d4 is empty and dark (95), e4 is empty and light (190).
        let h = board_to_image(&p.corners, 8).unwrap();
        let center = |s: u8| {
            let (row, col) = square_to_grid(s);
            let (x, y) = h.apply(col as f64 + 0.5, row as f64 + 0.5);
            img[y.round() as usize * 640 + x.round() as usize] as i32
        };
        assert!((center(27) - 95).abs() <= 2, "d4 = {}", center(27));
        assert!((center(28) - 190).abs() <= 2, "e4 = {}", center(28));
        // A corner of the image is outside the frame: background range.
        assert!((60..=180).contains(&img[0]), "background = {}", img[0]);
    }

    #[test]
    fn same_seed_same_image_and_brightness_shifts() {
        let board = startpos_board();
        let p = default_params(320, 240);
        let a = render(&board, &p);
        let b = render(&board, &p);
        assert_eq!(a, b);
        let bright = SynthParams { brightness: 8.0, ..default_params(320, 240) };
        let c = render(&board, &bright);
        let diff: i32 = a.iter().zip(&c).map(|(&x, &y)| y as i32 - x as i32).sum();
        assert!(diff > 0);
    }

    #[test]
    fn hand_blob_paints_gray_150() {
        let mut img = vec![0u8; 100 * 100];
        render_hand(&mut img, 100, 100, 50.0, 50.0, 10.0);
        assert_eq!(img[50 * 100 + 50], 150);
        assert_eq!(img[0], 0);
        assert_eq!(img[50 * 100 + 50 + 9], 150);
        assert_eq!(img[50 * 100 + 50 + 12], 0);
    }
}
