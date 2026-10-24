//! Scene stability gate.
//!
//! Decides when the scene has been still long enough to be worth analysing.
//! Each pushed frame is reduced to a small thumbnail (box-averaged cell means);
//! the mean absolute difference to the previous thumbnail is the "motion".
//! Motion above a threshold means the scene is moving; a run of low-motion
//! frames lasting `settle_ms` means it has settled.
//!
//! Everything is allocation-free per push after construction, so the gate can
//! run on every camera frame.
//! See docs/specs/cg-vision.md (Part B).
use std::time::{Duration, Instant};

pub struct GateConfig {
    pub thumb_w: u32,
    pub thumb_h: u32,
    pub motion_threshold: f32,
    /// A single thumbnail cell changing by more than this (gray levels) also
    /// counts as motion, so a localized change (one piece appearing between
    /// two frames) restarts the settle clock even though the mean barely moves.
    pub local_threshold: f32,
    pub settle_ms: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { thumb_w: 64, thumb_h: 36, motion_threshold: 3.0, local_threshold: 20.0, settle_ms: 300 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateState {
    Moving,
    Settling,
    /// `fresh` is true for exactly ONE call: the first frame after the scene settles.
    Stable { fresh: bool },
}

pub struct StabilityGate {
    cfg: GateConfig,
    /// Thumbnail of the most recent push (`thumb_w * thumb_h` cell means).
    prev: Vec<f32>,
    /// Scratch for the thumbnail being built; swapped with `prev` after each push.
    cur: Vec<f32>,
    /// Column-sum scratch for `thumbnail`, sized to the input width on first use.
    col: Vec<u32>,
    /// True once `prev` holds a thumbnail of the current input size.
    has_prev: bool,
    /// Input size the history belongs to. A different size resets the gate.
    dims: (u32, u32),
    motion: f32,
    /// Largest single-cell difference of the last push.
    local: f32,
    /// Start of the current still period, `None` while moving.
    still_since: Option<Instant>,
    /// Whether `Stable` was already reported for the current still period.
    announced: bool,
}

impl StabilityGate {
    pub fn new(cfg: GateConfig) -> Self {
        let n = (cfg.thumb_w * cfg.thumb_h) as usize;
        StabilityGate {
            cfg,
            prev: vec![0.0; n],
            cur: vec![0.0; n],
            col: Vec::new(),
            has_prev: false,
            dims: (0, 0),
            motion: 0.0,
            local: 0.0,
            still_since: None,
            announced: false,
        }
    }

    /// Feed one frame. `gray` is Luma8 with `width * height` pixels (any size:
    /// a full frame or a warped board). `at` is the frame time.
    pub fn push(&mut self, gray: &[u8], width: u32, height: u32, at: Instant) -> GateState {
        assert!(
            gray.len() >= width as usize * height as usize,
            "gate: buffer holds {} bytes, need {}x{}",
            gray.len(),
            width,
            height
        );
        if (width, height) != self.dims {
            self.reset();
            self.dims = (width, height);
        }

        self.col.resize(width as usize, 0);
        thumbnail(gray, width, height, self.cfg.thumb_w, self.cfg.thumb_h, &mut self.col, &mut self.cur);
        (self.motion, self.local) = if self.has_prev { abs_diff_stats(&self.cur, &self.prev) } else { (0.0, 0.0) };
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.has_prev = true;

        if self.motion > self.cfg.motion_threshold || self.local > self.cfg.local_threshold {
            self.still_since = None;
            self.announced = false;
            return GateState::Moving;
        }
        // Still this frame. The first still frame starts the clock.
        let since = match self.still_since {
            Some(t) => t,
            None => {
                self.still_since = Some(at);
                return GateState::Settling;
            }
        };
        if at.saturating_duration_since(since) < Duration::from_millis(self.cfg.settle_ms as u64) {
            return GateState::Settling;
        }
        let fresh = !self.announced;
        self.announced = true;
        GateState::Stable { fresh }
    }

    /// Mean absolute thumbnail difference of the last push (gray levels).
    pub fn motion(&self) -> f32 {
        self.motion
    }

    /// Forget history (e.g. after relocating the board). The next push returns `Settling`.
    pub fn reset(&mut self) {
        self.has_prev = false;
        self.motion = 0.0;
        self.local = 0.0;
        self.still_since = None;
        self.announced = false;
    }
}

/// Box-average `src` (Luma8, `w × h`) into `out` (`tw × th` cell means, row-major).
///
/// Cell `i` covers source columns `[i*w/tw, (i+1)*w/tw)`, so sizes that are not
/// exact multiples are handled: every source pixel is used and cells differ in
/// size by at most one pixel. `col` is scratch of length `w`. Column sums are
/// accumulated row by row first (a simple loop the compiler vectorises), then
/// reduced into cells.
fn thumbnail(src: &[u8], w: u32, h: u32, tw: u32, th: u32, col: &mut [u32], out: &mut [f32]) {
    let (w, h, tw, th) = (w as usize, h as usize, tw as usize, th as usize);
    // Range of source indices `[start, end)` that falls into cell `i` of `n` cells.
    let range = |i: usize, n: usize, t: usize| -> (usize, usize) {
        let start = (i * n / t).min(n - 1);
        let end = ((i + 1) * n / t).max(start + 1).min(n);
        (start, end)
    };
    for j in 0..th {
        let (y0, y1) = range(j, h, th);
        col.iter_mut().for_each(|c| *c = 0);
        for y in y0..y1 {
            let row = &src[y * w..(y + 1) * w];
            for (c, &p) in col.iter_mut().zip(row) {
                *c += p as u32;
            }
        }
        for i in 0..tw {
            let (x0, x1) = range(i, w, tw);
            let sum: u32 = col[x0..x1].iter().sum();
            let count = ((x1 - x0) * (y1 - y0)) as f32;
            out[j * tw + i] = sum as f32 / count;
        }
    }
}

/// (mean, max) absolute difference between two thumbnails.
fn abs_diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (sum, max) = a.iter().zip(b).fold((0f32, 0f32), |(s, m), (x, y)| {
        let d = (x - y).abs();
        (s + d, m.max(d))
    });
    (sum / a.len() as f32, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const W: u32 = 320;
    const H: u32 = 180;

    /// Deterministic xorshift64* so tests need no `rand` crate.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A smooth gradient, so thumbnail cells are not flat.
    fn scene() -> Vec<u8> {
        (0..W * H)
            .map(|i| {
                let (x, y) = (i % W, i / W);
                (60 + x * 100 / W + y * 60 / H) as u8
            })
            .collect()
    }

    fn frame_at(k: u64, step_ms: u64, start: Instant) -> Instant {
        start + Duration::from_millis(k * step_ms)
    }

    #[test]
    fn localized_change_counts_as_motion() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let mut img = scene();
        let start = Instant::now();
        for k in 0..8 {
            gate.push(&img, W, H, frame_at(k, 66, start));
        }
        // A 20x20 dark blob appears: ~0.7% of the image, mean motion tiny.
        for y in 80..100 {
            for x in 150..170 {
                img[(y * W + x) as usize] = 10;
            }
        }
        assert_eq!(gate.push(&img, W, H, frame_at(8, 66, start)), GateState::Moving);
        assert!(gate.motion() < 3.0);
    }

    #[test]
    fn static_scene_settles_once_then_stays_stable() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let img = scene();
        let start = Instant::now();
        let mut states = Vec::new();
        for k in 0..10 {
            states.push(gate.push(&img, W, H, frame_at(k, 66, start)));
        }
        // 66 ms steps: settle_ms = 300 is reached at k = 5 (330 ms).
        for s in &states[..5] {
            assert_eq!(*s, GateState::Settling);
        }
        assert_eq!(states[5], GateState::Stable { fresh: true });
        for s in &states[6..] {
            assert_eq!(*s, GateState::Stable { fresh: false });
        }
        assert!(gate.motion() < 1e-6);
    }

    #[test]
    fn moving_blob_is_moving() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let start = Instant::now();
        // A 100x100 bright blob moving 30 px per frame.
        for k in 0..8u32 {
            let x0 = 10 + 30 * k;
            let img: Vec<u8> = (0..W * H)
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    if x >= x0 && x < x0 + 100 && (40..140).contains(&y) {
                        220
                    } else {
                        40
                    }
                })
                .collect();
            let st = gate.push(&img, W, H, frame_at(k as u64, 66, start));
            if k == 0 {
                // The first frame has no history, so it cannot be moving yet.
                assert_eq!(st, GateState::Settling);
                continue;
            }
            assert_eq!(st, GateState::Moving, "frame {k}, motion {}", gate.motion());
            assert!(gate.motion() > 3.0);
        }
    }

    #[test]
    fn small_noise_stays_stable() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let base = scene();
        let start = Instant::now();
        let mut s = 99u64;
        let mut max_motion = 0f32;
        let mut stable_seen = false;
        for k in 0..12u64 {
            // Uniform noise in [-3, 3] on every pixel, different each frame.
            let img: Vec<u8> = base
                .iter()
                .map(|&p| {
                    let n = (next(&mut s) % 7) as i32 - 3;
                    (p as i32 + n).clamp(0, 255) as u8
                })
                .collect();
            let st = gate.push(&img, W, H, frame_at(k, 66, start));
            assert_ne!(st, GateState::Moving, "frame {k} motion {}", gate.motion());
            max_motion = max_motion.max(gate.motion());
            if matches!(st, GateState::Stable { .. }) {
                stable_seen = true;
            }
        }
        assert!(stable_seen, "noise should still count as settled");
        println!("gate: max motion under +-3 noise = {max_motion:.3}");
    }

    #[test]
    fn stable_fresh_fires_again_after_motion() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let still = scene();
        let moved: Vec<u8> = still.iter().map(|&p| p.saturating_add(60)).collect();
        let start = Instant::now();

        let mut fresh_count = 0;
        // Still period, motion, still period.
        for (t, img) in [&still, &still, &still, &still, &still, &still, &moved, &still, &still, &still, &still, &still, &still].into_iter().enumerate()
        {
            let st = gate.push(img, W, H, frame_at(t as u64, 100, start));
            if st == (GateState::Stable { fresh: true }) {
                fresh_count += 1;
            }

        }
        assert_eq!(fresh_count, 2);
    }

    #[test]
    fn reset_and_size_change_restart_settling() {
        let mut gate = StabilityGate::new(GateConfig::default());
        let img = scene();
        let start = Instant::now();
        assert_eq!(gate.push(&img, W, H, frame_at(0, 100, start)), GateState::Settling);
        assert!(matches!(gate.push(&img, W, H, frame_at(5, 100, start)), GateState::Stable { fresh: true }));

        gate.reset();
        assert_eq!(gate.push(&img, W, H, frame_at(6, 100, start)), GateState::Settling);

        // A different input size is a new scene: settling again, no motion measured.
        let small = scene();
        let st = gate.push(&small[..(160 * 90) as usize], 160, 90, frame_at(20, 100, start));
        assert_eq!(st, GateState::Settling);
        assert_eq!(gate.motion(), 0.0);
    }

    #[test]
    fn thumbnail_handles_non_multiple_sizes() {
        // 5x3 source into 2x2 cells: every pixel is used exactly once.
        let src: Vec<u8> = (0..15).map(|i| (i * 10) as u8).collect();
        let mut out = vec![0f32; 4];
        let mut col = vec![0u32; 5];
        thumbnail(&src, 5, 3, 2, 2, &mut col, &mut out);
        // Columns [0,2) and [2,5); rows [0,1) and [1,3). Source values are 10 * (5*row + col).
        assert_eq!(out[0], 10.0 / 2.0); // row 0, cols 0-1 (values 0 and 10)
        assert_eq!(out[1], (20 + 30 + 40) as f32 / 3.0); // row 0, cols 2-4
        assert_eq!(out[2], (50 + 60 + 100 + 110) as f32 / 4.0); // rows 1-2, cols 0-1
        assert_eq!(out[3], (70 + 80 + 90 + 120 + 130 + 140) as f32 / 6.0); // rows 1-2, cols 2-4
    }

    #[test]
    fn gate_cost_1280x720() {
        let (w, h) = (1280u32, 720u32);
        let mut s = 11u64;
        let frames: Vec<Vec<u8>> = (0..4)
            .map(|_| (0..w * h).map(|_| (next(&mut s) % 256) as u8).collect())
            .collect();
        let mut gate = StabilityGate::new(GateConfig::default());
        let start = Instant::now();
        gate.push(&frames[0], w, h, start);
        let iters = 200;
        let t0 = Instant::now();
        for k in 0..iters {
            gate.push(&frames[k % frames.len()], w, h, start);
        }
        let per = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("gate push 1280x720: {per:.3} ms per frame (spec target < 0.2 ms, release)");
    }
}
