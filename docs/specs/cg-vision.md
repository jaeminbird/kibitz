# cg-vision spec

Pure Rust, no image crates, no `unsafe`. Only deps: `cg-core`, `shakmaty` (0.30). Built at opt-level 3 even in tests (see workspace Cargo.toml).

Already written; **do not change its public API**: `src/geometry.rs` (`Homography`, `GrayImage`, `WarpedBoard`, `warp_board`, `square_to_grid`, `grid_to_square`, `board_to_image`). Read it first.

Conventions:
- A **square index** is shakmaty's: a1=0 … h8=63 (`shakmaty::Square as u8`, `Square::new(u32)`).
- A **warped board** is always from White's view: row 0 = rank 8, col 0 = file a.
- `Corners` (from cg-core) are in board order `[a8, h8, h1, a1]`, in image pixels.

`lib.rs` already declares the modules and re-exports. Make every re-exported name exist with exactly these signatures.

---

## Part A (agent "vision-a"): `features.rs`, `infer.rs`, `synth.rs`

### features.rs
```rust
pub struct SquareStats {
    /// Per square: 4×4 grid of cell means over the inner 70% of the square, in globally-normalised units.
    pub cells: [[f32; 16]; 64],
    /// Per square: std-dev of the inner region (normalised units).
    pub std: [f32; 64],
}
pub fn square_stats(b: &WarpedBoard) -> SquareStats;
pub fn square_diff(reference: &SquareStats, current: &SquareStats) -> [f32; 64];
```
- Global normalisation: compute the mean `μ` and std `σ` of the entire warped image, then use `(p-μ)/max(σ,1)` per pixel. This makes the features robust to auto-exposure changes.
- Inner region: drop 15% on each side of the square.
- `square_diff[s] = mean_k |cells_ref[s][k] - cells_cur[s][k]| + |std_ref[s] - std_cur[s]|`.
- It must be fast: `square_stats` on a 256×256 board in under 0.3 ms in release. Compute with integer sums, with no allocation beyond the output.

### infer.rs
```rust
pub struct InferConfig { pub min_change: f32, pub min_margin: f32, pub max_residual: f32 }
impl Default for InferConfig  // tune using synth tests; document the chosen values

pub struct Inference {
    /// Best legal move (if any square changed meaningfully).
    pub best: Option<shakmaty::Move>,
    pub best_uci: Option<String>,
    pub score: f32,
    /// best score minus the best score among moves with a DIFFERENT changed-square set.
    pub margin: f32,
    /// Sum of positive excess on squares NOT in best's changed set.
    pub residual: f32,
    /// Number of squares whose diff exceeds the threshold.
    pub changed: u32,
    /// Top 5 (uci, score), best first, one entry per distinct changed-square set.
    pub ranked: Vec<(String, f32)>,
    /// best.is_some() && margin >= min_margin && residual <= max_residual
    pub accept: bool,
    /// When !accept: "no change" | "ambiguous" | "unexplained change" | "no legal moves"
    pub reason: Option<&'static str>,
}

/// Squares whose appearance changes when `mv` is played in `pos`:
/// from, to; castling: king from/to AND rook from/to (shakmaty represents castling as king→rook; handle it);
/// en passant: also the captured pawn's square. Sorted, deduplicated.
pub fn changed_squares(pos: &shakmaty::Chess, mv: &shakmaty::Move) -> Vec<u8>;

pub fn infer_move(pos: &shakmaty::Chess, diff: &[f32; 64], cfg: &InferConfig) -> Inference;
```
Algorithm:
1. Threshold `t = max(cfg.min_change, 2.5 * median(diff))`. `e_s = diff[s] - t`. `changed = #{e_s > 0}`. If `changed == 0`, return `reason = "no change"`.
2. For each legal move, `score(m) = Σ_{s ∈ C(m)} e_s`. Unchanged squares in C(m) count negatively, which correctly penalises e.g. castling when only the king moved.
3. Group moves with an identical C(m) (promotions to Q/R/B/N). Within a group, pick the queen promotion, else the first.
4. best = max group. margin = best − second-best group (or `best.score` if only one group). residual = Σ max(0, e_s) for s ∉ C(best).
5. Use the UCI string in standard notation (`e1g1` for castling): `mv.to_uci(CastlingMode::Standard).to_string()`.

### synth.rs — synthetic renderer (used by tests and the `bench` command)
```rust
pub struct SynthParams {
    pub width: u32, pub height: u32,
    /// Board corners in board order [a8,h8,h1,a1], image pixels.
    pub corners: cg_core::Corners,
    /// Gaussian-ish pixel noise std-dev in gray levels.
    pub noise: f32,
    /// Added to every pixel (simulates exposure change).
    pub brightness: f32,
    pub seed: u64,
}
pub fn render(board: &shakmaty::Board, p: &SynthParams) -> Vec<u8>;   // Luma8, width*height
pub fn default_params(width: u32, height: u32) -> SynthParams;          // board ~60% of height, mild perspective, noise 3
pub fn render_hand(img: &mut [u8], width: u32, height: u32, cx: f32, cy: f32, r: f32); // skin-tone (gray 150) ellipse blob
```
- Background: low-frequency deterministic texture from the seed (sum of a few sines). Board border: a 0.4-square-wide frame of gray 120 around the playing area.
- Squares: light = 190, dark = 95 (a1 is dark).
- Pieces: filled disc of radius 0.36 sq, white pieces 240, black pieces 25, plus an inner ring whose radius depends on the role (pawn 0.10 … king 0.30), drawn with the opposite tone. Pieces of different roles therefore look different.
- Render by inverse mapping: for each image pixel, map through the inverse homography (image → board coordinates, see `board_to_image` and invert by solving `from_points` the other way) and shade.
- Deterministic PRNG (xorshift64*), no `rand` crate. Noise is the sum of 4 uniform values, scaled.

### Tests (in each module, `#[cfg(test)]`)
- `changed_squares`: e2e4 → {e2,e4}; O-O for White → {e1,f1,g1,h1}; en passant → 3 squares; promotion.
- End-to-end with synth: for ≥ 20 positions from a fixed random walk of legal moves (seeded), render before/after with **different seeds/noise** and brightness +8, warp both with `warp_board(.., sq_px=32)`, run `square_stats` → `square_diff` → `infer_move`. The played move must be best and `accept` must be true in ≥ 95% of cases, and **never accept a wrong move**. Include castling, a capture, en passant if reachable (construct from FEN), and promotion.
- "No move" case: same position, different noise → `reason == "no change"`, not accepted.
- With a hand blob covering part of the board in the "after" image → must not accept a wrong move.
- Write a test that prints timing (`square_stats` + `infer_move`) in a release run, but don't assert on time.

---

## Part B (agent "vision-b"): `gate.rs`, `track.rs`

### gate.rs — stability gate (replaces the cloud pre-check)
```rust
pub struct GateConfig { pub thumb_w: u32, pub thumb_h: u32, pub motion_threshold: f32, pub settle_ms: u32 }
impl Default for GateConfig  // 64, 36, 3.0, 300

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateState {
    Moving,
    Settling,
    /// `fresh` is true for exactly ONE call: the first frame after the scene settles.
    Stable { fresh: bool },
}

pub struct StabilityGate { /* … */ }
impl StabilityGate {
    pub fn new(cfg: GateConfig) -> Self;
    /// `gray` is any Luma8 image (full frame or a warped board). `at` = frame time.
    pub fn push(&mut self, gray: &[u8], width: u32, height: u32, at: std::time::Instant) -> GateState;
    /// Mean absolute thumbnail difference of the last push (gray levels).
    pub fn motion(&self) -> f32;
    /// Forget history (e.g. after relocating the board). Next push returns Settling.
    pub fn reset(&mut self);
}
```
- Thumbnail by box-averaging into `thumb_w × thumb_h` (handle sizes that aren't exact multiples). motion = mean |thumb − prev_thumb|.
- motion > threshold → Moving (still_since = None). Otherwise, if still_since is None, set it to `at` and return Settling; if `at − still_since < settle_ms`, Settling; else Stable, with `fresh` true only the first time in this still period.
- If the input dimensions change, reset.
- Cost: under 0.2 ms for 1280×720 (release).
- Tests: a static image sequence becomes Stable{fresh:true} once after settle_ms (use synthetic `Instant`s: `start + Duration::from_millis(k*66)`), then Stable{fresh:false}. A moving blob gives Moving. Small noise (±3) stays stable.

### track.rs — corner tracker + board validation
```rust
pub struct TrackerConfig {
    pub coarse_scale: u32,   // 8: downscale factor for global shift search
    pub search_px: i32,      // 64: max global shift per frame, full-res pixels
    pub patch_px: u32,       // 48: corner template size, full-res
    pub refine_px: i32,      // 10: per-corner search radius after the global shift
    pub min_checker: f32,    // 0.35
}
impl Default for TrackerConfig

pub struct CornerTracker { /* … */ }
impl CornerTracker {
    pub fn new(cfg: TrackerConfig) -> Self;
    /// Adopt `corners` as truth for this frame. Stores corner templates (kept for life → no drift)
    /// and a coarse reference image (updated after every successful track).
    pub fn reset(&mut self, gray: &[u8], width: u32, height: u32, corners: cg_core::Corners);
    /// Follow the board into this frame. None = not found in this frame only: corners, templates
    /// and the coarse reference are kept, and the next call searches more widely.
    pub fn track(&mut self, gray: &[u8], width: u32, height: u32) -> Option<cg_core::Corners>;
    /// Consecutive failed frames since the last accepted one.
    pub fn misses(&self) -> u32;
    /// Drop all tracking state. Called by the app when it declares the board lost.
    pub fn clear(&mut self);
    pub fn corners(&self) -> Option<cg_core::Corners>;
    /// Last checker score computed by track().
    pub fn last_score(&self) -> f32;
}

/// How board-like a warped image is, in [-1, 1]. Uses the OUTER BAND of each square
/// (the 15% border ring, where the square color shows even when a piece is on it):
/// m_s = band mean, p_s = +1 for light squares, −1 for dark (a1 dark),
/// score = Σ p_s (m_s − mean(m)) / Σ |m_s − mean(m)|.
pub fn checker_score(b: &WarpedBoard) -> f32;
```
Algorithm for `track`:
1. **Global shift:** downscale the current frame by `coarse_scale` (box filter). SAD-match the bounding box of the previous corners (shrunk 10%) from the coarse reference against the current coarse frame, over shifts within ±search_px/coarse_scale. Then refine at scale 2 within ±2·(coarse_scale/2) of the coarse result. Result: (dx, dy) in full-res pixels.
2. **Per-corner refine:** for each corner, search the template (taken at `reset`, centred on the corner) in the current frame within ±refine_px around `corner_at_reset + (current_shift_total)`. Use zero-mean NCC. Accept a corner only if NCC ≥ 0.5; otherwise use the shifted position.
3. **Validate:** warp with the new corners at sq_px = 16, compute `checker_score`. If < min_checker, this frame is a miss: return None and keep all state.
4. On success, update the coarse reference to this frame, store the new corners, and reset the miss count.
5. **Misses widen the search.** After 1 and 2 consecutive misses the global-shift window is ±2·search_px and ±4·search_px. From the 3rd consecutive miss on, every attempt searches the whole frame: the best few SAD minima at 4× coarse_scale (plus the ±4·search_px window) are each refined as in steps 1–2, and the candidate with the highest `checker_score` wins. Any re-acquisition after a miss also needs ≥ 3 corners whose template NCC reaches 0.5, so a board seen one square off cannot be adopted.
6. **Frame-size change** (stream renegotiation, window resize, DPI change): before step 1, scale the stored corners, reset corners and velocities by the per-axis size ratio, sample the corner templates at the new scale (the sample grid is resized to cover the same part of the scene), and resample the coarse and scale-2 references to the new sizes. Then track as usual. A size change never drops the track.
7. The tracker never drops its state on its own. `clear()` does that; cg-app calls it when tracking has failed for longer than its grace period (1.5 s).
- Cost target: `track` under 4 ms on 1280×720 (release). Corner templates stay fixed (no drift); only the coarse reference updates.
- Tests (use `crate::synth` if part A has landed, or else your own simple renderer: checkerboard + frame on a textured background):
  - `checker_score` > 0.6 for a correctly warped synthetic board with the start position. It is negative for corners rotated by one place (a1 ↔ a8 swapped orientation flips the colors).
  - Translate the camera by up to 40 px per frame over 10 frames: the tracked corners stay within 2 px of the truth.
  - A small rotation (2°/frame for 5 frames) stays within 3 px.
  - A frame with the board fully replaced by background → None, with the corners kept; the next frame showing the board tracks again.
  - 1, 2, 3 and 10 garbage frames (noise, flat gray, tone-inverted board) inserted into a drifting sequence, with the camera jumping meanwhile: each garbage frame → None, and tracking resumes within 2 px on the first good frame.
  - The same scene at 1280×720 → 640×360 → 1920×1080 → 1280×720 → 960×720 (aspect change) → 1280×720: tracked within 2 px at every size. A miss at a new size keeps the corners, rescaled.

---

## Rules for both agents
- Edit only your own module files (plus, if strictly necessary, add `#[cfg(test)]` helpers in your own files). Do not edit `geometry.rs`, `lib.rs`, or other crates. If you believe the contract must change, stop and report why instead of changing it.
- Use your own target dir to avoid lock contention: `CARGO_TARGET_DIR=target/agent-<yourname> cargo test -p cg-vision --release`.
- If another agent's module doesn't compile yet, comment out nothing. Instead, temporarily test with `cargo test -p cg-vision --release <your_module>`. If the crate as a whole fails because the other module is missing, create a minimal stub for the missing file **only if it doesn't exist**, containing `todo!()` bodies with the exact signatures above, and say so in your report.
- `cargo clippy -p cg-vision` should be clean of warnings in your files.
