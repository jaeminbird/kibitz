//! Test rig: drive the real [`VisionPipeline`] with synthetic frames
//! (`cg_vision::synth::render`) and a fake clock, as `bench` does.

use std::time::{Duration, Instant};

use cg_core::{Corners, Frame};
use cg_vision::synth;
use cg_vision::geometry::board_to_image;
use shakmaty::{Board, Chess, Square};

use crate::vision::{Outcome, VisionPipeline};

pub const W: u32 = 640;
pub const H: u32 = 480;
const FRAME_DT: Duration = Duration::from_millis(66);
/// Frames to hold a scene so the gate (300 ms settle) reports Stable.
pub const SETTLE_FRAMES: usize = 8;

pub struct Rig {
    pub vision: VisionPipeline,
    corners: Corners,
    seq: u64,
    clock: Instant,
}

impl Rig {
    /// A pipeline calibrated (as a manual click would) on `board`.
    pub fn new(board: &Board) -> Rig {
        let corners = synth::default_params(W, H).corners;
        let mut rig = Rig { vision: VisionPipeline::new(300), corners, seq: 0, clock: Instant::now() };
        let first = rig.frame(Some(board), None);
        rig.vision.set_corners(&first, corners);
        rig
    }

    /// One frame of `board` (None: the board is out of view), optionally with
    /// a hand blob of radius `r` px centred on square `hand`.
    pub fn frame(&mut self, board: Option<&Board>, hand: Option<(Square, f32)>) -> Frame {
        let mut p = synth::default_params(W, H);
        p.corners = self.corners;
        p.seed = self.seq * 7919 + 1;
        self.seq += 1;
        self.clock += FRAME_DT;
        let mut gray = match board {
            Some(b) => synth::render(b, &p),
            None => vec![128; (W * H) as usize],
        };
        if let Some((sq, r)) = hand {
            let to_img = board_to_image(&self.corners, 8).expect("valid corners");
            let (row, col) = cg_vision::square_to_grid(u8::from(sq));
            let (x, y) = to_img.apply(col as f64 + 0.5, row as f64 + 0.5);
            synth::render_hand(&mut gray, W, H, x as f32, y as f32, r);
        }
        let mut f = Frame::from_gray(self.seq, W, H, gray);
        f.at = self.clock;
        f
    }

    /// Show the scene for `n` frames and return every outcome.
    pub fn hold(&mut self, board: Option<&Board>, hand: Option<(Square, f32)>, n: usize, pos: &Chess) -> Vec<Outcome> {
        (0..n)
            .map(|_| {
                let f = self.frame(board, hand);
                self.vision.process(&f, pos)
            })
            .collect()
    }

    /// Show the scene until it settles; return the first settled outcome, if any.
    pub fn settle(&mut self, board: Option<&Board>, hand: Option<(Square, f32)>, pos: &Chess) -> Option<Outcome> {
        self.hold(board, hand, SETTLE_FRAMES, pos).into_iter().find(|o| {
            !matches!(o, Outcome::Moving | Outcome::Settling | Outcome::Searching | Outcome::NoBoard)
        })
    }
}
