//! Board corner tracker and checkerboard validation.
//!
//! Once the board has been located, the corners are followed frame to frame
//! in three steps, cheapest first:
//!
//! 1. **Global shift.** The previous frame is kept as a two-level pyramid
//!    (scale 8 for a coarse search, scale 2 for refinement). The bounding box
//!    of the board is matched by sum of absolute differences over a window of
//!    shifts. This handles camera translation.
//! 2. **Per-corner refine.** Each corner has a zero-mean template cut at
//!    `reset` time (never updated, so it cannot drift). It is matched by
//!    normalised cross-correlation (NCC) near where the shift predicts it, with
//!    a parabolic sub-pixel fit on the NCC peak. This handles small rotations.
//! 3. **Validate.** The board is re-warped and must still look like a
//!    checkerboard (`checker_score`), otherwise the track is lost.
//!
//! See docs/specs/cg-vision.md (Part B).
use cg_core::Corners;

use crate::geometry::{warp_board, WarpedBoard};

/// Square size used when validating a tracked board. Small on purpose: the
/// check only needs the colour pattern, and it runs on every frame.
const VALIDATE_SQ_PX: u32 = 16;

/// Minimum NCC for a corner refinement to be accepted.
const MIN_NCC: f64 = 0.5;

/// How many extra search windows a corner may chase when its peak sits on the window edge.
const MAX_CHASE: usize = 2;

pub struct TrackerConfig {
    pub coarse_scale: u32,
    pub search_px: i32,
    pub patch_px: u32,
    pub refine_px: i32,
    pub min_checker: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig { coarse_scale: 8, search_px: 64, patch_px: 48, refine_px: 10, min_checker: 0.35 }
    }
}

pub struct CornerTracker {
    cfg: TrackerConfig,
    /// Frame size the state belongs to. A different size loses the track.
    dims: (u32, u32),
    /// Corners of the last accepted frame; `None` when not tracking.
    corners: Option<Corners>,
    /// One template per corner, cut at `reset` time. Kept for life.
    templates: Vec<Template>,
    /// Scale-2 image of the last accepted frame (refinement reference).
    half_ref: Plane,
    /// Scale-`coarse_scale` image of the last accepted frame (coarse reference).
    coarse_ref: Plane,
    /// Per-corner motion of the last accepted frame beyond the global shift (a rotation, for
    /// example). Assumed to continue, so the refine search starts closer to the truth.
    velocity: [[f32; 2]; 4],
    /// Corners at `reset`; the templates were cut around these.
    reset_corners: Corners,
    last_score: f32,
}

impl CornerTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        CornerTracker {
            cfg,
            dims: (0, 0),
            corners: None,
            templates: Vec::new(),
            half_ref: Plane::default(),
            coarse_ref: Plane::default(),
            velocity: [[0.0; 2]; 4],
            reset_corners: Corners([[0.0; 2]; 4]),
            last_score: 0.0,
        }
    }

    /// Adopt `corners` as truth for this frame. Stores corner templates (kept for life → no drift)
    /// and a coarse reference image (updated after every successful track).
    pub fn reset(&mut self, gray: &[u8], width: u32, height: u32, corners: Corners) {
        assert!(gray.len() >= width as usize * height as usize, "tracker: short buffer");
        let (w, h) = (width as usize, height as usize);
        let (half, coarse) = pyramid(gray, w, h, self.coarse_scale());
        let step = self.template_step();
        self.templates = corners
            .0
            .iter()
            .map(|&p| Template::new(gray, w, h, p, self.cfg.patch_px as usize, step))
            .collect();
        self.dims = (width, height);
        self.half_ref = half;
        self.coarse_ref = coarse;
        self.corners = Some(corners);
        self.reset_corners = corners;
        self.velocity = [[0.0; 2]; 4];
        self.last_score = 0.0;
    }

    /// Follow the board into this frame. None = lost (also clears state).
    pub fn track(&mut self, gray: &[u8], width: u32, height: u32) -> Option<Corners> {
        self.last_score = 0.0;
        let prev = self.corners?;
        if (width, height) != self.dims || self.templates.len() != 4 {
            self.clear();
            return None;
        }
        assert!(gray.len() >= width as usize * height as usize, "tracker: short buffer");
        let (w, h) = (width as usize, height as usize);
        let c = self.coarse_scale();
        let (half, coarse) = pyramid(gray, w, h, c);
        if half.w < 8 || half.h < 8 {
            self.clear();
            return None;
        }

        // 1. Global shift of the whole board, in full-resolution pixels.
        let Some((gx, gy)) = self.global_shift(&half, &coarse, &prev, c) else {
            self.clear();
            return None;
        };

        // 2. Per-corner refinement around the predicted position: previous corner + global
        //    shift + the corner's residual motion from the last frame.
        let g = [gx as f32, gy as f32];
        // The templates are rotated by the board's rotation since reset, measured on the last accepted corners.
        let phi = similarity_angle(&self.reset_corners, &prev);
        let mut out = [[0f32; 2]; 4];
        for (i, o) in out.iter_mut().enumerate() {
            let p = prev.0[i];
            let v = self.velocity[i];
            let pred = [p[0] + g[0] + v[0], p[1] + g[1] + v[1]];
            let sampled = self.templates[i].sample(phi);
            *o = self.refine_corner(gray, w, h, &sampled, pred);
            self.velocity[i] = [o[0] - p[0] - g[0], o[1] - p[1] - g[1]];
        }

        // 3. The new corners must still show a checkerboard.
        let Some(board) = warp_board(gray, width, height, &Corners(out), VALIDATE_SQ_PX) else {
            self.clear();
            return None;
        };
        let score = checker_score(&board);
        self.last_score = score;
        if score < self.cfg.min_checker {
            self.clear();
            return None;
        }

        // Accept: move the coarse reference forward; templates stay as they are.
        self.corners = Some(Corners(out));
        self.half_ref = half;
        self.coarse_ref = coarse;
        Some(Corners(out))
    }

    pub fn corners(&self) -> Option<Corners> {
        self.corners
    }

    /// Last checker score computed by track().
    pub fn last_score(&self) -> f32 {
        self.last_score
    }

    fn coarse_scale(&self) -> usize {
        self.cfg.coarse_scale.max(2) as usize
    }

    /// Sample spacing inside a corner template. Keeps templates to about 24×24 samples.
    fn template_step(&self) -> usize {
        (self.cfg.patch_px as usize / 24).max(1)
    }

    fn clear(&mut self) {
        self.corners = None;
        self.templates.clear();
        self.half_ref = Plane::default();
        self.coarse_ref = Plane::default();
        self.velocity = [[0.0; 2]; 4];
    }

    /// Translation of the board since the last accepted frame, in full-resolution pixels.
    fn global_shift(&self, half: &Plane, coarse: &Plane, prev: &Corners, c: usize) -> Option<(i32, i32)> {
        // Match the shrunk bounding box, so the board's background margin does not dominate.
        let bbox = shrunk_bbox(prev, 0.9);

        // Coarse search, in coarse pixels, over ±search_px.
        let rect_c = plane_rect(bbox, c as f32, coarse.w, coarse.h)?;
        let radius_c = (self.cfg.search_px / c as i32).max(1);
        let (sx, sy) = sad_search(&self.coarse_ref, coarse, rect_c, (0, 0), radius_c)?;

        // Refinement at scale 2, around the coarse estimate, ±c half-pixels (covers the coarse quantisation).
        let rect_h = plane_rect(bbox, 2.0, half.w, half.h)?;
        let c = c as i32;
        let center = ((sx * c) / 2, (sy * c) / 2);
        let (hx, hy) = sad_search(&self.half_ref, half, rect_h, center, c)?;
        Some((hx * 2, hy * 2))
    }

    /// Best NCC match of `t` near `pred`. Returns `pred` unchanged when no match reaches `MIN_NCC`.
    ///
    /// The search window is `±refine_px` around the prediction. If the peak lands on the window edge
    /// the window is re-centred on it (up to `MAX_CHASE` times), so a fast corner is not clipped.
    fn refine_corner(&self, gray: &[u8], w: usize, h: usize, t: &Sampled, pred: [f32; 2]) -> [f32; 2] {
        let r = self.cfg.refine_px;
        let (mut cx, mut cy) = (pred[0].round() as i32, pred[1].round() as i32);
        let mut best = (f64::NEG_INFINITY, cx, cy);
        for _ in 0..=MAX_CHASE {
            best = search_window(gray, w, h, t, cx, cy, r);
            let (ox, oy) = (best.1 - cx, best.2 - cy);
            if ox.abs() < r && oy.abs() < r {
                break;
            }
            (cx, cy) = (best.1, best.2);
        }
        if best.0 < MIN_NCC {
            return pred;
        }
        // Parabolic sub-pixel fit through the peak and its neighbours on each axis.
        let (bx, by) = (best.1, best.2);
        let fit = |minus: f64, zero: f64, plus: f64| -> f32 {
            let den = minus - 2.0 * zero + plus;
            if den < 0.0 {
                (0.5 * (minus - plus) / den).clamp(-0.5, 0.5) as f32
            } else {
                0.0
            }
        };
        let dx = fit(ncc(gray, w, h, t, bx - 1, by), best.0, ncc(gray, w, h, t, bx + 1, by));
        let dy = fit(ncc(gray, w, h, t, bx, by - 1), best.0, ncc(gray, w, h, t, bx, by + 1));
        // The template was cut around a rounded corner; add back its fractional offset.
        [bx as f32 + dx + t.frac[0], by as f32 + dy + t.frac[1]]
    }
}

/// Owned 8-bit image used for the search pyramid.
#[derive(Clone, Debug, Default)]
struct Plane {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

/// Box-average `src` (`w × h`) down by an integer factor `f`. Output size is `w/f × h/f`.
fn downscale(src: &[u8], w: usize, h: usize, f: usize) -> Plane {
    let (ow, oh) = (w / f, h / f);
    let area = (f * f) as u32;
    let mut data = vec![0u8; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = 0u32;
            for dy in 0..f {
                let start = (oy * f + dy) * w + ox * f;
                sum += src[start..start + f].iter().map(|&p| p as u32).sum::<u32>();
            }
            data[oy * ow + ox] = ((sum + area / 2) / area) as u8;
        }
    }
    Plane { w: ow, h: oh, data }
}

/// The scale-2 and scale-`c` images of a frame. The coarse one is built from the
/// scale-2 image when `c` is even, which saves a pass over the full frame.
fn pyramid(gray: &[u8], w: usize, h: usize, c: usize) -> (Plane, Plane) {
    let half = downscale(gray, w, h, 2);
    let coarse = if c.is_multiple_of(2) { downscale(&half.data, half.w, half.h, c / 2) } else { downscale(gray, w, h, c) };
    (half, coarse)
}

/// Bounding box `[x0, y0, x1, y1]` of the corners, scaled about its centre by `factor`.
fn shrunk_bbox(corners: &Corners, factor: f32) -> [f32; 4] {
    let xs = corners.0.map(|p| p[0]);
    let ys = corners.0.map(|p| p[1]);
    let (x0, x1) = (xs.iter().copied().fold(f32::INFINITY, f32::min), xs.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let (y0, y1) = (ys.iter().copied().fold(f32::INFINITY, f32::min), ys.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (hx, hy) = ((x1 - x0) / 2.0 * factor, (y1 - y0) / 2.0 * factor);
    [cx - hx, cy - hy, cx + hx, cy + hy]
}

/// Half-open pixel rectangle `(x0, y0, x1, y1)` inside a plane.
type Rect = (usize, usize, usize, usize);

/// Map a full-resolution box onto a plane that is `scale` times smaller, clamped to the plane.
fn plane_rect(bbox: [f32; 4], scale: f32, pw: usize, ph: usize) -> Option<Rect> {
    if pw < 2 || ph < 2 {
        return None;
    }
    let clamp_x = |v: f32| (v / scale).clamp(0.0, pw as f32 - 1.0);
    let clamp_y = |v: f32| (v / scale).clamp(0.0, ph as f32 - 1.0);
    let x0 = clamp_x(bbox[0]).floor() as usize;
    let y0 = clamp_y(bbox[1]).floor() as usize;
    let x1 = (clamp_x(bbox[2]).ceil() as usize).max(x0 + 1);
    let y1 = (clamp_y(bbox[3]).ceil() as usize).max(y0 + 1);
    if x1 - x0 < 4 || y1 - y0 < 4 {
        return None;
    }
    Some((x0, y0, x1.min(pw), y1.min(ph)))
}

/// Find the shift that best matches `rect` of `reference` in `current`.
/// Searches `center ± radius` (per axis) and returns the shift with the lowest mean absolute difference.
/// Compares a subsampled grid (at most about 64 samples per side) to bound the cost.
fn sad_search(reference: &Plane, current: &Plane, rect: Rect, center: (i32, i32), radius: i32) -> Option<(i32, i32)> {
    let (x0, y0, x1, y1) = rect;
    let (rw, rh) = (x1 - x0, y1 - y0);
    let step = (rw.max(rh) / 64).max(1);
    let mut best: Option<(u64, i32, i32)> = None;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (sx, sy) = (center.0 + dx, center.1 + dy);
            let (cx0, cy0) = (x0 as i64 + sx as i64, y0 as i64 + sy as i64);
            if cx0 < 0 || cy0 < 0 || cx0 as usize + rw > current.w || cy0 as usize + rh > current.h {
                continue;
            }
            let (cx0, cy0) = (cx0 as usize, cy0 as usize);
            let mut sad = 0u64;
            for ty in (0..rh).step_by(step) {
                let rrow = &reference.data[(y0 + ty) * reference.w + x0..];
                let crow = &current.data[(cy0 + ty) * current.w + cx0..];
                for tx in (0..rw).step_by(step) {
                    sad += rrow[tx].abs_diff(crow[tx]) as u64;
                }
            }
            if best.is_none_or(|(b, _, _)| sad < b) {
                best = Some((sad, sx, sy));
            }
        }
    }
    best.map(|(_, sx, sy)| (sx, sy))
}

/// A corner's appearance as it looked at `reset`: the raw pixels around the corner.
/// Kept for life; it is never re-cut, so the track cannot drift. Rotation is handled by
/// sampling it rotated (see [`Template::sample`]).
#[derive(Clone, Debug)]
struct Template {
    /// Samples per side, and the spacing between them.
    n: usize,
    step: usize,
    /// Distance from the template centre to its first sample.
    half: usize,
    /// Raw `side × side` patch centred on the rounded corner, big enough to rotate the sample grid.
    raw: Vec<f32>,
    side: usize,
    /// Offset of the true corner from the integer template centre.
    frac: [f32; 2],
}

/// Zero-mean samples of a template at one board rotation, ready for NCC.
#[derive(Clone, Debug)]
struct Sampled {
    n: usize,
    step: usize,
    /// Distance from the template centre to its first sample.
    half: usize,
    /// Samples minus their mean, row-major.
    samples: Vec<f64>,
    /// Sum of squared samples (the template's energy, for NCC).
    /// Offset of the true corner from the integer template centre.
    frac: [f32; 2],
    norm: f64,
}

impl Template {
    fn new(gray: &[u8], w: usize, h: usize, corner: [f32; 2], patch: usize, step: usize) -> Template {
        let n = (patch / step).max(2);
        let half = n * step / 2;
        // Enough raw pixels to cover the sample grid at any rotation (corner distance is half·√2).
        let rad = (half as f32 * std::f32::consts::SQRT_2).ceil() as usize + 2;
        let side = 2 * rad + 1;
        let (cx, cy) = (corner[0].round() as i64, corner[1].round() as i64);
        let mut raw = vec![0f32; side * side];
        for j in 0..side {
            for i in 0..side {
                // Clamp at the frame edge; a template that runs off the frame is still usable.
                let x = (cx + i as i64 - rad as i64).clamp(0, w as i64 - 1) as usize;
                let y = (cy + j as i64 - rad as i64).clamp(0, h as i64 - 1) as usize;
                raw[j * side + i] = gray[y * w + x] as f32;
            }
        }
        let frac = [corner[0] - cx as f32, corner[1] - cy as f32];
        Template { n, step, half, raw, side, frac }
    }

    /// The template as it appears when the board is rotated by `phi` (radians, image
    /// coordinates) about the corner. Each sample is read from the raw patch at the
    /// position it came from, `R(-phi)·offset`, using bilinear interpolation.
    fn sample(&self, phi: f32) -> Sampled {
        let (s, c) = (-phi).sin_cos();
        let rad = (self.side / 2) as f32;
        let mut samples = Vec::with_capacity(self.n * self.n);
        for j in 0..self.n {
            for i in 0..self.n {
                let ox = (i * self.step) as f32 - self.half as f32;
                let oy = (j * self.step) as f32 - self.half as f32;
                let (qx, qy) = (c * ox - s * oy, s * ox + c * oy);
                samples.push(bilinear(&self.raw, self.side, rad + qx, rad + qy) as f64);
            }
        }
        let mean = samples.iter().sum::<f64>() / (self.n * self.n) as f64;
        samples.iter_mut().for_each(|v| *v -= mean);
        let norm = samples.iter().map(|v| v * v).sum();
        Sampled { n: self.n, step: self.step, half: self.half, samples, norm, frac: self.frac }
    }
}

/// Bilinear sample of a square `side × side` image at (`x`, `y`), clamped to the image.
fn bilinear(img: &[f32], side: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, side as f32 - 1.0);
    let y = y.clamp(0.0, side as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(side - 1), (y0 + 1).min(side - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = img[y0 * side + x0] * (1.0 - fx) + img[y0 * side + x1] * fx;
    let bot = img[y1 * side + x0] * (1.0 - fx) + img[y1 * side + x1] * fx;
    top * (1.0 - fy) + bot * fy
}

/// Rotation (radians, image coordinates, same convention as [`Homography`]) that takes the
/// board `from` to the board `to`, estimated from the four edges.
fn similarity_angle(from: &Corners, to: &Corners) -> f32 {
    let (mut cross, mut dot) = (0f64, 0f64);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let r = [(from.0[j][0] - from.0[i][0]) as f64, (from.0[j][1] - from.0[i][1]) as f64];
        let t = [(to.0[j][0] - to.0[i][0]) as f64, (to.0[j][1] - to.0[i][1]) as f64];
        cross += r[0] * t[1] - r[1] * t[0];
        dot += r[0] * t[0] + r[1] * t[1];
    }
    cross.atan2(dot) as f32
}

/// Best NCC over the `±r` window around (`cx`, `cy`). Returns (ncc, x, y).
fn search_window(gray: &[u8], w: usize, h: usize, t: &Sampled, cx: i32, cy: i32, r: i32) -> (f64, i32, i32) {
    let mut best = (f64::NEG_INFINITY, cx, cy);
    for oy in -r..=r {
        for ox in -r..=r {
            let v = ncc(gray, w, h, t, cx + ox, cy + oy);
            if v > best.0 {
                best = (v, cx + ox, cy + oy);
            }
        }
    }
    best
}

/// NCC of `t` with its centre at integer pixel (`px`, `py`) in the frame.
/// Returns -1 when the template would run off the frame.
fn ncc(gray: &[u8], w: usize, h: usize, t: &Sampled, px: i32, py: i32) -> f64 {
    let span = (t.n - 1) * t.step;
    let x0 = px as i64 - t.half as i64;
    let y0 = py as i64 - t.half as i64;
    if x0 < 0 || y0 < 0 || x0 + span as i64 >= w as i64 || y0 + span as i64 >= h as i64 {
        return -1.0;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let (mut sp, mut spp, mut spt) = (0f64, 0f64, 0f64);
    for j in 0..t.n {
        let row = (y0 + j * t.step) * w + x0;
        for i in 0..t.n {
            let p = gray[row + i * t.step] as f64;
            sp += p;
            spp += p * p;
            spt += p * t.samples[j * t.n + i];
        }
    }
    // Samples are zero-mean, so Σ(p-μ)t = Σpt.
    let nn = (t.n * t.n) as f64;
    let var = spp - sp * sp / nn;
    if var <= 1e-9 || t.norm <= 1e-9 {
        return 0.0;
    }
    spt / (var * t.norm).sqrt()
}

/// How board-like a warped image is, in [-1, 1]. Uses the OUTER BAND of each square
/// (the 15% border ring, where the square color shows even when a piece is on it):
/// m_s = band mean, p_s = +1 for light squares, −1 for dark (a1 dark),
/// score = Σ p_s (m_s − mean(m)) / Σ |m_s − mean(m)|.
///
/// A correct board scores near +1. Rotated by a square, the colours flip and the score
/// goes negative. A flat or unstructured image scores near 0.
pub fn checker_score(b: &WarpedBoard) -> f32 {
    let s = b.sq_px;
    let size = b.size() as usize;
    // Band width: 15% of a square, at least one pixel.
    let band = ((s as f32 * 0.15).round() as u32).max(1);
    let mut means = [0f32; 64];
    for (sq, mean) in means.iter_mut().enumerate() {
        let (x0, y0, _, _) = b.square_rect(sq as u8);
        let (mut sum, mut n) = (0u64, 0u32);
        for dy in 0..s {
            let edge_row = dy < band || dy >= s - band;
            let row = (y0 as usize + dy as usize) * size + x0 as usize;
            for dx in 0..s {
                if edge_row || dx < band || dx >= s - band {
                    sum += b.img.data[row + dx as usize] as u64;
                    n += 1;
                }
            }
        }
        *mean = sum as f32 / n as f32;
    }

    let overall = means.iter().sum::<f32>() / 64.0;
    let (mut num, mut den) = (0f32, 0f32);
    for (sq, &m) in means.iter().enumerate() {
        let d = m - overall;
        // Shakmaty index: file = sq % 8, rank = sq / 8. a1 (0) is dark, so light squares have odd file+rank.
        let light = (sq % 8 + sq / 8) % 2 == 1;
        num += if light { d } else { -d };
        den += d.abs();
    }
    if den < 1e-6 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Homography;
    use shakmaty::{Board, Chess, Color, Position, Role, Square};
    use std::time::Instant;

    const W: u32 = 1280;
    const H: u32 = 720;

    /// Start position board.
    fn start_board() -> Board {
        Chess::default().board().clone()
    }

    /// A board of about 470 px, with mild perspective: [a8, h8, h1, a1].
    fn start_corners() -> Corners {
        Corners([[430.0, 130.0], [850.0, 125.0], [900.0, 600.0], [380.0, 605.0]])
    }

    /// Test-only renderer, independent of `crate::synth`.
    /// Textured background, an 8×8 board with a 0.4-square frame around the playing area,
    /// and simple discs for pieces (white 240, black 25) with an inner ring of the opposite tone.
    /// Each pixel is mapped through the inverse homography to board units.
    struct Scene<'a> {
        /// `None` renders background only (the board has left the frame).
        board: Option<&'a Board>,
        /// Playing-area corners in image pixels, [a8, h8, h1, a1].
        corners: Corners,
        /// Offset of the background texture, so a moving camera also moves the background.
        bg_offset: (f32, f32),
        /// Rotation of the background about `bg_center`, in radians (camera roll).
        bg_angle: f32,
        bg_center: [f32; 2],
        /// Gaussian-like pixel noise std-dev in gray levels.
        noise: f32,
        noise_seed: u64,
    }

    /// xorshift64*, so tests need no `rand` crate.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(state: &mut u64) -> f32 {
        (next(state) >> 11) as f32 / (1u64 << 53) as f32
    }

    fn role_ring_radius(role: Role) -> f32 {
        match role {
            Role::Pawn => 0.10,
            Role::Knight => 0.14,
            Role::Bishop => 0.18,
            Role::Rook => 0.22,
            Role::Queen => 0.26,
            Role::King => 0.30,
        }
    }

    /// Gray value of the playing area at board coordinates `(u, v)` in squares, with (0,0) = a8.
    fn board_tone(board: &Board, u: f32, v: f32) -> f32 {
        if !(0.0..8.0).contains(&u) || !(0.0..8.0).contains(&v) {
            return 120.0; // the frame
        }
        let (col, row) = (u.floor(), v.floor());
        let (file, rank) = (col as u32, 7 - row as u32);
        let light = (file + rank) % 2 == 1;
        let base = if light { 190.0 } else { 95.0 };
        let Some(piece) = board.piece_at(Square::new(file + 8 * rank)) else {
            return base;
        };
        let (dx, dy) = (u - col - 0.5, v - row - 0.5);
        let r = (dx * dx + dy * dy).sqrt();
        let (own, opposite) = if piece.color == Color::White { (240.0, 25.0) } else { (25.0, 240.0) };
        if r < 0.36 {
            if (r - role_ring_radius(piece.role)).abs() < 0.035 {
                opposite
            } else {
                own
            }
        } else {
            base
        }
    }

    fn render(scene: &Scene) -> Vec<u8> {
        // Image pixels → board units. Source: the image corners; destination: the unit square.
        let src = scene.corners.0.map(|[x, y]| [x as f64, y as f64]);
        let inv = Homography::from_points(src, [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
            .expect("non-degenerate corners");
        // Fixed texture phases: sum of a few low-frequency sines.
        let (p1, p2, p3) = (0.7f32, 2.1f32, 4.3f32);
        let background = |x: f32, y: f32| -> f32 {
            128.0
                + 45.0 * (0.021 * x + 0.013 * y + p1).sin()
                + 30.0 * (0.047 * x - 0.031 * y + p2).sin()
                + 18.0 * (0.11 * x + 0.07 * y + p3).sin()
        };
        let (sn, cs) = (-scene.bg_angle).sin_cos();
        let mut rng = scene.noise_seed | 1;
        let (w, h) = (W as usize, H as usize);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (u, v) = inv.apply(x as f64 + 0.5, y as f64 + 0.5);
                let (u, v) = (u as f32, v as f32);
                let mut val = match scene.board {
                    Some(b) if (-0.4..8.4).contains(&u) && (-0.4..8.4).contains(&v) => board_tone(b, u, v),
                    _ => {
                        // Screen point → world point: undo the camera roll, then the offset.
                        let (dx, dy) = (x as f32 - scene.bg_center[0], y as f32 - scene.bg_center[1]);
                        let wx = cs * dx - sn * dy + scene.bg_center[0] - scene.bg_offset.0;
                        let wy = sn * dx + cs * dy + scene.bg_center[1] - scene.bg_offset.1;
                        background(wx, wy)
                    }
                };
                if scene.noise > 0.0 {
                    // Sum of 4 uniforms minus 2 has variance 1/3, so scale by noise / sqrt(1/3).
                    let n = unit(&mut rng) + unit(&mut rng) + unit(&mut rng) + unit(&mut rng) - 2.0;
                    val += n * scene.noise / 0.577;
                }
                out[y * w + x] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    fn shift(c: Corners, dx: f32, dy: f32) -> Corners {
        Corners(c.0.map(|[x, y]| [x + dx, y + dy]))
    }

    fn rotate(c: Corners, center: [f32; 2], angle: f32) -> Corners {
        let (s, co) = angle.sin_cos();
        Corners(c.0.map(|[x, y]| {
            let (dx, dy) = (x - center[0], y - center[1]);
            [center[0] + dx * co - dy * s, center[1] + dx * s + dy * co]
        }))
    }

    fn centroid(c: Corners) -> [f32; 2] {
        let n = 4.0;
        let (sx, sy) = c.0.iter().fold((0.0, 0.0), |(a, b), p| (a + p[0], b + p[1]));
        [sx / n, sy / n]
    }

    fn max_corner_error(got: Corners, truth: Corners) -> f32 {
        got.0
            .iter()
            .zip(truth.0.iter())
            .map(|(g, t)| ((g[0] - t[0]).powi(2) + (g[1] - t[1]).powi(2)).sqrt())
            .fold(0.0, f32::max)
    }

    fn render_board(board: &Board, corners: Corners, noise_seed: u64) -> Vec<u8> {
        render(&Scene { board: Some(board), corners, bg_offset: (0.0, 0.0), bg_angle: 0.0, bg_center: [0.0, 0.0], noise: 2.0, noise_seed })
    }

    #[test]
    fn checker_score_start_position_is_positive() {
        let board = start_board();
        let c = start_corners();
        let frame = render_board(&board, c, 1);
        let warped = warp_board(&frame, W, H, &c, 32).unwrap();
        let s = checker_score(&warped);
        println!("checker_score start position: {s:.3}");
        assert!(s > 0.6, "score {s}");
    }

    #[test]
    fn checker_score_negative_when_rotated_one_place() {
        let board = start_board();
        let c = start_corners();
        let frame = render_board(&board, c, 2);
        // Corner labels shifted by one: the board is seen rotated 90°, so the colours flip.
        let rotated = Corners([c.0[1], c.0[2], c.0[3], c.0[0]]);
        let warped = warp_board(&frame, W, H, &rotated, 32).unwrap();
        let s = checker_score(&warped);
        println!("checker_score rotated one place: {s:.3}");
        assert!(s < 0.0, "score {s}");
    }

    #[test]
    fn checker_score_is_near_zero_on_background() {
        let c = start_corners();
        let frame = render(&Scene { board: None, corners: c, bg_offset: (0.0, 0.0), bg_angle: 0.0, bg_center: [0.0, 0.0], noise: 2.0, noise_seed: 3 });
        let warped = warp_board(&frame, W, H, &c, 32).unwrap();
        let s = checker_score(&warped);
        println!("checker_score background: {s:.3}");
        assert!(s.abs() < 0.35, "score {s}");
    }

    #[test]
    fn tracks_camera_translation_within_2px() {
        let board = start_board();
        let base = start_corners();
        // Per-frame camera moves, up to 40 px, with the cumulative offset kept inside the frame.
        let steps = [
            (40.0, 10.0),
            (-35.0, 20.0),
            (30.0, -25.0),
            (-40.0, -15.0),
            (20.0, 30.0),
            (40.0, -10.0),
            (-30.0, 25.0),
            (35.0, -30.0),
            (-25.0, 15.0),
            (10.0, -5.0),
        ];
        let mut tracker = CornerTracker::new(TrackerConfig::default());
        tracker.reset(&render_board(&board, base, 10), W, H, base);
        let (mut cx, mut cy) = (0f32, 0f32);
        let mut worst = 0f32;
        for (k, &(dx, dy)) in steps.iter().enumerate() {
            cx += dx;
            cy += dy;
            let truth = shift(base, cx, cy);
            // The camera moved, so the background moves with the board.
            let frame = render(&Scene {
                board: Some(&board),
                corners: truth,
                bg_offset: (cx, cy),
                bg_angle: 0.0,
                bg_center: [0.0, 0.0],
                noise: 2.0,
                noise_seed: 10 + k as u64,
            });
            let got = tracker.track(&frame, W, H).unwrap_or_else(|| panic!("frame {k}: lost track"));
            let err = max_corner_error(got, truth);
            println!("translation frame {k}: offset ({cx:+.0},{cy:+.0}) max corner error {err:.2} px, score {:.2}", tracker.last_score());
            worst = worst.max(err);
            assert!(err <= 2.0, "frame {k}: error {err:.2}");
        }
        println!("translation: worst corner error {worst:.2} px over {} frames", steps.len());
    }

    #[test]
    fn tracks_small_rotation_within_3px() {
        let board = start_board();
        let base = start_corners();
        let center = centroid(base);
        // A camera roll: the whole picture rotates about the board centre, 2° per frame.
        // (The corner templates cover only the board and its frame, so a static background would
        // give the same result.)
        let mut tracker = CornerTracker::new(TrackerConfig::default());
        tracker.reset(&render_board(&board, base, 20), W, H, base);
        let mut worst = 0f32;
        for k in 1..=5u64 {
            let angle = (2.0 * k as f32).to_radians();
            let truth = rotate(base, center, angle);
            let frame = render(&Scene {
                board: Some(&board),
                corners: truth,
                bg_offset: (0.0, 0.0),
                bg_angle: angle,
                bg_center: center,
                noise: 2.0,
                noise_seed: 20 + k,
            });
            let got = tracker.track(&frame, W, H).unwrap_or_else(|| panic!("frame {k}: lost track"));
            let err = max_corner_error(got, truth);
            println!("rotation frame {k}: {} deg, max corner error {err:.2} px", 2 * k);
            worst = worst.max(err);
            assert!(err <= 3.0, "frame {k}: error {err:.2}");
        }
        println!("rotation: worst corner error {worst:.2} px");
    }

    #[test]
    fn background_only_frame_loses_track() {
        let board = start_board();
        let base = start_corners();
        let mut tracker = CornerTracker::new(TrackerConfig::default());
        tracker.reset(&render_board(&board, base, 30), W, H, base);
        let frame = render(&Scene { board: None, corners: base, bg_offset: (0.0, 0.0), bg_angle: 0.0, bg_center: [0.0, 0.0], noise: 2.0, noise_seed: 31 });
        assert!(tracker.track(&frame, W, H).is_none());
        assert!(tracker.corners().is_none());
        assert!(tracker.last_score() < 0.35);
    }

    #[test]
    fn untracked_or_resized_input_is_lost() {
        let board = start_board();
        let base = start_corners();
        let frame = render_board(&board, base, 40);
        let mut tracker = CornerTracker::new(TrackerConfig::default());
        assert!(tracker.track(&frame, W, H).is_none(), "no reset yet");
        tracker.reset(&frame, W, H, base);
        assert!(tracker.track(&frame[..(640 * 360) as usize], 640, 360).is_none(), "size change");
        assert!(tracker.corners().is_none());
    }

    #[test]
    fn track_cost_1280x720() {
        let board = start_board();
        let base = start_corners();
        let frames: Vec<Vec<u8>> = (0..6)
            .map(|k| {
                let c = shift(base, 6.0 * (k % 2) as f32, 4.0 * (k % 3) as f32);
                render_board(&board, c, 50 + k)
            })
            .collect();
        let mut tracker = CornerTracker::new(TrackerConfig::default());
        tracker.reset(&frames[0], W, H, base);
        let iters = 60;
        let t0 = Instant::now();
        for k in 0..iters {
            let got = tracker.track(&frames[k % frames.len()], W, H);
            assert!(got.is_some(), "iteration {k} lost track");
        }
        let per = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("track 1280x720: {per:.3} ms per frame (spec target < 4 ms, release)");
    }
}

