//! Per-frame vision pipeline: track the board, rectify it, gate on stability,
//! and (on a freshly settled board) infer which legal move was played.
//! Shared by the game actor and the `bench` command.

use std::time::{Duration, Instant};

use cg_core::{Corners, Frame};
use cg_vision::{
    infer_move, square_diff, square_stats, warp_board, CornerTracker, GateConfig, GateState, InferConfig, Inference,
    SquareStats, StabilityGate, TrackerConfig,
};
use shakmaty::Chess;

const SQ_PX: u32 = 32;
/// How long tracking may fail before we declare the board lost.
const LOST_AFTER: Duration = Duration::from_millis(1500);

pub enum Outcome {
    /// No board location yet (needs calibration or an LLM locate).
    NoBoard,
    /// Tracking failed this frame (within the grace period).
    Searching,
    /// Tracking failed for longer than the grace period; corners were dropped.
    Lost,
    Moving,
    Settling,
    /// The scene settled and a new reference board was captured (after calibration/undo).
    Baseline,
    /// Settled scene, nothing new to read.
    Stable,
    /// First settled frame after movement: here is what changed.
    Read { inference: Inference, stats: Box<SquareStats> },
}

pub struct VisionPipeline {
    tracker: CornerTracker,
    gate: StabilityGate,
    infer_cfg: InferConfig,
    corners: Option<Corners>,
    /// Frame size `corners` are expressed in.
    dims: (u32, u32),
    reference: Option<SquareStats>,
    failing_since: Option<Instant>,
    pub last_frame_ms: f32,
}

impl VisionPipeline {
    pub fn new(settle_ms: u32) -> VisionPipeline {
        VisionPipeline {
            tracker: CornerTracker::new(TrackerConfig::default()),
            gate: StabilityGate::new(GateConfig { settle_ms, ..GateConfig::default() }),
            infer_cfg: InferConfig::default(),
            corners: None,
            dims: (0, 0),
            reference: None,
            failing_since: None,
            last_frame_ms: 0.0,
        }
    }

    pub fn min_margin(&self) -> f32 {
        self.infer_cfg.min_margin
    }

    pub fn max_residual(&self) -> f32 {
        self.infer_cfg.max_residual
    }

    pub fn corners(&self) -> Option<Corners> {
        self.corners
    }

    /// Adopt board corners located in `frame` (manual click or LLM).
    pub fn set_corners(&mut self, frame: &Frame, corners: Corners) {
        self.tracker.reset(&frame.gray, frame.width, frame.height, corners);
        self.gate.reset();
        self.corners = Some(corners);
        self.dims = (frame.width, frame.height);
        self.reference = None;
        self.failing_since = None;
    }

    /// Drop the board location. This is the only place the tracker's state is cleared: a
    /// failed `track` keeps it, so the tracker can re-acquire within the `LOST_AFTER` grace period.
    pub fn forget_board(&mut self) {
        self.tracker.clear();
        self.corners = None;
        self.reference = None;
        self.failing_since = None;
    }

    /// The physical board may not match the reference any more (undo, manual
    /// move, new game): re-baseline on the next settled frame.
    pub fn rebaseline(&mut self) {
        self.reference = None;
    }

    /// Accept `stats` (from a Read) as the new reference after a commit.
    pub fn commit_reference(&mut self, stats: Box<SquareStats>) {
        self.reference = Some(*stats);
    }

    pub fn process(&mut self, frame: &Frame, pos: &Chess) -> Outcome {
        let t0 = Instant::now();
        let out = self.process_inner(frame, pos);
        self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
        out
    }

    fn process_inner(&mut self, frame: &Frame, pos: &Chess) -> Outcome {
        let Some(known) = self.corners else {
            return Outcome::NoBoard;
        };
        if (frame.width, frame.height) != self.dims {
            // The stream changed resolution (renegotiation, window resize, DPI change). Keep the
            // calibration in the new pixel grid; the tracker rescales its own state the same way.
            let (sx, sy) = (frame.width as f32 / self.dims.0.max(1) as f32, frame.height as f32 / self.dims.1.max(1) as f32);
            self.corners = Some(Corners(known.0.map(|[x, y]| [x * sx, y * sy])));
            self.dims = (frame.width, frame.height);
        }
        let Some(corners) = self.tracker.track(&frame.gray, frame.width, frame.height) else {
            // The tracker keeps its state and searches wider on each miss; give it LOST_AFTER to recover.
            let since = *self.failing_since.get_or_insert(frame.at);
            if frame.at.duration_since(since) >= LOST_AFTER {
                self.forget_board();
                return Outcome::Lost;
            }
            return Outcome::Searching;
        };
        self.failing_since = None;
        self.corners = Some(corners);

        let Some(board) = warp_board(&frame.gray, frame.width, frame.height, &corners, SQ_PX) else {
            return Outcome::Searching;
        };
        // Gate on the rectified board only: movement elsewhere in the room
        // (or the wearer's head, once tracked out) doesn't block reading.
        let size = board.size();
        match self.gate.push(&board.img.data, size, size, frame.at) {
            GateState::Moving => Outcome::Moving,
            GateState::Settling => Outcome::Settling,
            GateState::Stable { fresh } => {
                let Some(reference) = &self.reference else {
                    self.reference = Some(square_stats(&board));
                    return Outcome::Baseline;
                };
                if !fresh {
                    return Outcome::Stable;
                }
                let stats = square_stats(&board);
                let diff = square_diff(reference, &stats);
                let inference = infer_move(pos, &diff, &self.infer_cfg);
                Outcome::Read { inference, stats: Box::new(stats) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cg_vision::synth;
    use shakmaty::Position;

    const W: u32 = 640;
    const H: u32 = 480;
    const DT: Duration = Duration::from_millis(66);

    fn board_frame(seq: u64, at: Instant) -> (Frame, Corners) {
        board_frame_sized(seq, at, W, H)
    }

    fn board_frame_sized(seq: u64, at: Instant, w: u32, h: u32) -> (Frame, Corners) {
        let p = synth::default_params(w, h);
        let mut f = Frame::from_gray(seq, w, h, synth::render(Chess::default().board(), &p));
        f.at = at;
        (f, p.corners)
    }

    fn max_err(a: Corners, b: Corners) -> f32 {
        a.0.iter().zip(b.0.iter()).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).fold(0.0, f32::max)
    }

    #[test]
    fn resolution_change_rescales_corners_and_keeps_tracking() {
        let pos = Chess::default();
        let t0 = Instant::now();
        let mut vision = VisionPipeline::new(300);
        let (first, corners) = board_frame(1, t0);
        vision.set_corners(&first, corners);
        // A garbage frame at the new size: the published corners must already match that size.
        let mut garbage = Frame::from_gray(2, W / 2, H / 2, vec![128; (W * H / 4) as usize]);
        garbage.at = t0 + DT;
        assert!(matches!(vision.process(&garbage, &pos), Outcome::Searching));
        let half = Corners(corners.0.map(|[x, y]| [x * 0.5, y * 0.5]));
        assert!(max_err(vision.corners().unwrap(), half) < 1e-3);
        // The board at the new size: tracking continues, no loss and no re-locate needed.
        let (small, truth) = board_frame_sized(3, t0 + DT * 2, W / 2, H / 2);
        assert!(matches!(vision.process(&small, &pos), Outcome::Moving | Outcome::Settling));
        assert!(max_err(vision.corners().unwrap(), truth) <= 2.0);
        // And back up.
        let (big, truth) = board_frame_sized(4, t0 + DT * 3, W * 3 / 2, H * 3 / 2);
        assert!(matches!(vision.process(&big, &pos), Outcome::Moving | Outcome::Settling));
        assert!(max_err(vision.corners().unwrap(), truth) <= 2.0);
    }

    fn garbage_frame(seq: u64, at: Instant) -> Frame {
        let mut f = Frame::from_gray(seq, W, H, vec![128; (W * H) as usize]);
        f.at = at;
        f
    }

    #[test]
    fn tracking_resumes_within_grace_period() {
        let pos = Chess::default();
        let t0 = Instant::now();
        let mut vision = VisionPipeline::new(300);
        let (first, corners) = board_frame(1, t0);
        vision.set_corners(&first, corners);
        // About 0.7 s of unusable frames: inside LOST_AFTER, so the pipeline keeps searching.
        for k in 1..=10u32 {
            let out = vision.process(&garbage_frame(1 + k as u64, t0 + DT * k), &pos);
            assert!(matches!(out, Outcome::Searching), "garbage frame {k}");
            assert!(vision.corners().is_some(), "garbage frame {k} dropped the corners");
        }
        let (back, _) = board_frame(12, t0 + DT * 11);
        let out = vision.process(&back, &pos);
        assert!(matches!(out, Outcome::Moving | Outcome::Settling), "board did not resume");
        assert!(vision.failing_since.is_none());
    }

    #[test]
    fn board_is_lost_after_grace_period_and_tracker_cleared() {
        let pos = Chess::default();
        let t0 = Instant::now();
        let mut vision = VisionPipeline::new(300);
        let (first, corners) = board_frame(1, t0);
        vision.set_corners(&first, corners);
        let mut lost_at = None;
        for k in 1..=30u32 {
            if let Outcome::Lost = vision.process(&garbage_frame(1 + k as u64, t0 + DT * k), &pos) {
                lost_at = Some(k);
                break;
            }
        }
        let k = lost_at.expect("never declared lost");
        assert!(DT * (k - 1) >= LOST_AFTER, "lost after only {k} frames");
        assert!(vision.corners().is_none());
        // The tracker was cleared too, so it cannot silently bring the old corners back.
        assert!(vision.tracker.corners().is_none());
        let (back, _) = board_frame(100, t0 + DT * (k + 1));
        assert!(matches!(vision.process(&back, &pos), Outcome::NoBoard));
    }
}
