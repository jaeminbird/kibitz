//! Per-frame vision pipeline: track the board, rectify it, gate on stability,
//! and (on a freshly settled board) infer which legal move was played.
//! Shared by the game actor and the `bench` command.

use std::time::{Duration, Instant};

use cg_core::{Corners, Frame};
use cg_vision::{
    infer_move, infer_plies, square_diff, square_stats, warp_board, CornerTracker, GateConfig, GateState, InferConfig,
    Inference, PlyInference, SquareStats, StabilityGate, TrackerConfig,
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
    /// A read was missed since the reference (see `note_missed`), the plain
    /// one-move read failed on this settled frame, and `infer_plies` accepted
    /// a line (usually two plies): `plies` are the moves to commit, in order.
    Resync { plies: PlyInference, stats: Box<SquareStats> },
}

pub struct VisionPipeline {
    tracker: CornerTracker,
    gate: StabilityGate,
    infer_cfg: InferConfig,
    corners: Option<Corners>,
    /// Frame size `corners` are expressed in.
    dims: (u32, u32),
    reference: Option<SquareStats>,
    /// Since when a settled read may have gone uncommitted (ambiguous, no LLM
    /// answer), so the board may be a ply ahead of the reference.
    missed_since: Option<Instant>,
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
            missed_since: None,
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
        self.missed_since = None;
    }

    /// Accept `stats` (from a Read) as the new reference after a commit.
    pub fn commit_reference(&mut self, stats: Box<SquareStats>) {
        self.reference = Some(*stats);
        self.missed_since = None;
    }

    /// A settled read at `at` was not committed (ambiguous): keep the reference
    /// and check later reads over two plies against it.
    pub fn note_missed(&mut self, at: Instant) {
        self.missed_since.get_or_insert(at);
    }

    /// When the oldest uncommitted read happened, if one is still pending.
    pub fn missed_since(&self) -> Option<Instant> {
        self.missed_since
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
                if self.missed_since.is_some() && !inference.accept {
                    if inference.reason == Some("no change") {
                        // The board is back to the reference: nothing was missed.
                        self.missed_since = None;
                    } else {
                        let plies = infer_plies(pos, &diff, 2, &self.infer_cfg);
                        if plies.accept {
                            return Outcome::Resync { plies, stats: Box::new(stats) };
                        }
                    }
                }
                Outcome::Read { inference, stats: Box::new(stats) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game;
    use crate::testkit::Rig;
    use cg_vision::synth;
    use shakmaty::{Position, Square};

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

    fn play(pos: &Chess, ucis: &[&str]) -> Chess {
        let mut p = pos.clone();
        for u in ucis {
            let mv = game::parse_move(&p, u).expect("legal");
            p.play_unchecked(mv);
        }
        p
    }

    /// e2e4 is read while the pawn is still in the hand (only e2 changed:
    /// ambiguous), then e4 lands and Black answers e7e5 before the next read.
    fn missed_then_two_plies(rig: &mut Rig, pos: &Chess, note_missed: bool) -> Outcome {
        let mut lifted = pos.board().clone();
        lifted.discard_piece_at(Square::E2);
        match rig.settle(Some(&lifted), None, pos) {
            Some(Outcome::Read { inference, .. }) => {
                assert_eq!(inference.reason, Some("ambiguous"), "{inference:?}");
                if note_missed {
                    rig.vision.note_missed(Instant::now());
                }
            }
            _ => panic!("expected an ambiguous read"),
        }
        let after = play(pos, &["e2e4", "e7e5"]);
        rig.settle(Some(after.board()), None, pos).expect("settled")
    }

    #[test]
    fn missed_read_without_two_ply_freezes() {
        let pos = Chess::default();
        let mut rig = Rig::new(pos.board());
        assert!(matches!(rig.settle(Some(pos.board()), None, &pos), Some(Outcome::Baseline)));
        match missed_then_two_plies(&mut rig, &pos, false) {
            Outcome::Read { inference, .. } => assert!(!inference.accept, "one move cannot explain two"),
            _ => panic!("expected a plain read"),
        }
    }

    #[test]
    fn missed_read_is_resolved_over_two_plies() {
        let pos = Chess::default();
        let mut rig = Rig::new(pos.board());
        assert!(matches!(rig.settle(Some(pos.board()), None, &pos), Some(Outcome::Baseline)));
        match missed_then_two_plies(&mut rig, &pos, true) {
            Outcome::Resync { plies, stats } => {
                assert!(plies.accept);
                assert_eq!(plies.ucis, ["e2e4", "e7e5"]);
                rig.vision.commit_reference(stats);
            }
            _ => panic!("expected a two-ply resync"),
        }
        assert!(rig.vision.missed_since().is_none());
        // Reading continues from the new reference.
        let pos = play(&pos, &["e2e4", "e7e5"]);
        let after = play(&pos, &["g1f3"]);
        match rig.settle(Some(after.board()), None, &pos) {
            Some(Outcome::Read { inference, .. }) => {
                assert!(inference.accept);
                assert_eq!(inference.best_uci.as_deref(), Some("g1f3"));
            }
            _ => panic!("expected a read"),
        }
    }

    #[test]
    fn missed_read_clears_when_the_board_returns_to_the_reference() {
        let pos = Chess::default();
        let mut rig = Rig::new(pos.board());
        assert!(matches!(rig.settle(Some(pos.board()), None, &pos), Some(Outcome::Baseline)));
        rig.vision.note_missed(Instant::now());
        match rig.settle(Some(pos.board()), Some((Square::D5, 45.0)), &pos) {
            Some(Outcome::Read { inference, .. }) => assert!(!inference.accept),
            _ => panic!("expected a read"),
        }
        assert!(rig.vision.missed_since().is_some());
        match rig.settle(Some(pos.board()), None, &pos) {
            Some(Outcome::Read { inference, .. }) => assert_eq!(inference.reason, Some("no change")),
            _ => panic!("expected a read"),
        }
        assert!(rig.vision.missed_since().is_none());
    }
}
