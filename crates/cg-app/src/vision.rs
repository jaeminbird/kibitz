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
        self.reference = None;
        self.failing_since = None;
    }

    pub fn forget_board(&mut self) {
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
        if self.corners.is_none() {
            return Outcome::NoBoard;
        }
        let Some(corners) = self.tracker.track(&frame.gray, frame.width, frame.height) else {
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
