//! The game actor: the single owner of game state. Reacts to new frames,
//! controls, engine replies and LLM replies; never sleeps on the hot path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cg_capture::CaptureStatus;
use cg_core::{
    spoken_san, Bus, Control, Corners, EventKind, Frame, Latency, Side, Snapshot, Status, Suggestion,
};
use cg_engine::{Engine, EngineError, EngineReply, Position as EnginePos};
use cg_llm::{Cost, VisionLlm};
use cg_voice::Speaker;
use shakmaty::{Color, Position};
use tokio::sync::{mpsc, watch};

use crate::game::{self, GameState};
use crate::vision::{Outcome, VisionPipeline};

/// Retry the engine at most this often after a failure.
const ENGINE_RETRY: Duration = Duration::from_secs(1);
/// Minimum gap between LLM board-locate attempts.
const LOCATE_EVERY: Duration = Duration::from_secs(5);
/// Unexplained reads in a row before declaring lost track.
const UNEXPLAINED_LIMIT: u32 = 3;
/// How long a missed (uncommitted) read may stay unresolved before we ask the
/// user to enter the move.
const MISSED_ANNOUNCE: Duration = Duration::from_secs(4);

pub struct Config {
    pub my_color: Color,
    pub fen: Option<String>,
    pub settle_ms: u32,
    pub ponder: bool,
    pub max_cost: f64,
    pub run_dir: Option<PathBuf>,
}

pub struct Deps {
    pub bus: Bus,
    pub engine: Engine,
    pub llm: Option<Arc<VisionLlm>>,
    pub speaker: Arc<dyn Speaker>,
}

type LocateResult = Result<(Option<[[f32; 2]; 4]>, Cost), cg_llm::LlmError>;

enum Msg {
    Engine { epoch: u64, result: Result<EngineReply, EngineError>, asked_at: Instant },
    Located { frame: Frame, result: LocateResult },
    Picked { epoch: u64, frame_at: Instant, result: Result<(Option<String>, Cost), cg_llm::LlmError> },
}

struct Actor {
    cfg: Config,
    bus: Bus,
    engine: Engine,
    llm: Option<Arc<VisionLlm>>,
    speaker: Arc<dyn Speaker>,
    tx: mpsc::UnboundedSender<Msg>,

    game: GameState,
    /// Bumped on every change to the move list; stale async replies are dropped.
    epoch: u64,
    vision: VisionPipeline,
    paused: bool,
    source_missing: bool,
    ended: bool,
    lost: bool,
    unexplained: UnexplainedRun,
    /// "Didn't catch that move" was said for the pending missed read.
    missed_spoken: bool,

    suggestion: Option<Suggestion>,
    last_reply: Option<EngineReply>,
    engine_inflight: bool,
    engine_failed_at: Option<Instant>,
    engine_error_spoken: bool,

    llm_busy: bool,
    last_locate: Option<Instant>,
    spent: f64,
    budget_spoken: bool,

    latest_frame: Option<Frame>,
    /// When the board settled for the move being answered (for total latency).
    settled_at: Option<Instant>,
    latency: Latency,
    dirty: bool,
}

pub async fn run(
    cfg: Config,
    deps: Deps,
    mut frames: watch::Receiver<Option<Frame>>,
    mut controls: mpsc::Receiver<Control>,
    mut capture: mpsc::UnboundedReceiver<CaptureStatus>,
) -> anyhow::Result<()> {
    let game = GameState::new(cfg.my_color, cfg.fen.as_deref()).map_err(anyhow::Error::msg)?;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut a = Actor {
        vision: VisionPipeline::new(cfg.settle_ms),
        cfg,
        bus: deps.bus,
        engine: deps.engine,
        llm: deps.llm,
        speaker: deps.speaker,
        tx,
        game,
        epoch: 0,
        paused: false,
        source_missing: false,
        ended: false,
        lost: false,
        unexplained: UnexplainedRun::default(),
        missed_spoken: false,
        suggestion: None,
        last_reply: None,
        engine_inflight: false,
        engine_failed_at: None,
        engine_error_spoken: false,
        llm_busy: false,
        last_locate: None,
        spent: 0.0,
        budget_spoken: false,
        latest_frame: None,
        settled_at: None,
        latency: Latency::default(),
        dirty: true,
    };
    a.bus.emit(EventKind::NewGame { color: color_name(a.cfg.my_color).into(), fen: a.game.fen() });
    a.maybe_request_suggestion();
    a.publish();

    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut frames_open = true;
    loop {
        tokio::select! {
            changed = frames.changed(), if frames_open => match changed {
                Ok(()) => {
                    let frame = frames.borrow_and_update().clone();
                    if let Some(f) = frame { a.on_frame(f); }
                }
                Err(_) => frames_open = false,
            },
            Some(c) = controls.recv() => a.on_control(c),
            Some(m) = rx.recv() => a.on_msg(m),
            Some(s) = capture.recv() => a.on_capture(s),
            _ = tick.tick() => a.on_tick(),
            _ = tokio::signal::ctrl_c() => break,
        }
        if a.dirty {
            a.publish();
        }
    }
    a.engine.stop().await;
    Ok(())
}

impl Actor {
    // ---------------------------------------------------------------- frames

    fn on_frame(&mut self, frame: Frame) {
        self.source_missing = false;
        self.latest_frame = Some(frame.clone());
        if self.paused || self.game.outcome().is_some() {
            return;
        }
        let pos = self.game.position().clone();
        match self.vision.process(&frame, &pos) {
            Outcome::NoBoard => self.maybe_locate(&frame),
            Outcome::Lost => {
                self.set_lost("board not visible");
            }
            Outcome::Searching | Outcome::Moving | Outcome::Settling | Outcome::Stable => {}
            Outcome::Baseline => {
                self.lost = false;
                self.unexplained.reset();
                self.dirty = true;
            }
            Outcome::Read { inference, stats } => self.on_read(&frame, inference, stats),
            Outcome::Resync { plies, stats } => self.on_resync(&frame, plies, stats),
        }
    }

    /// A line of plies explains the board since the reference (a read was missed): commit them in order.
    fn on_resync(&mut self, frame: &Frame, plies: cg_vision::PlyInference, stats: Box<cg_vision::SquareStats>) {
        self.bus.emit(EventKind::Frame {
            seq: frame.seq,
            changed_squares: plies.changed,
            best: Some(plies.ucis.join(" ")),
            margin: Some(plies.margin),
            skipped: None,
            vision_ms: self.vision.last_frame_ms,
        });
        self.unexplained.reset();
        self.lost = false;
        self.vision.commit_reference(stats);
        self.settled_at = Some(frame.at);
        for mv in plies.moves {
            if self.game.outcome().is_some() {
                break;
            }
            self.commit(mv, "vision", Some(plies.margin), Some(frame.at));
        }
    }

    fn on_read(&mut self, frame: &Frame, inf: cg_vision::Inference, stats: Box<cg_vision::SquareStats>) {
        let vision_ms = self.vision.last_frame_ms;
        self.bus.emit(EventKind::Frame {
            seq: frame.seq,
            changed_squares: inf.changed,
            best: inf.best_uci.clone(),
            margin: inf.best.as_ref().map(|_| inf.margin),
            skipped: inf.reason.map(String::from),
            vision_ms,
        });
        // Only consecutive unexplained reads count: any other settled read resets the run.
        let lost_run = self.unexplained.record(inf.reason == Some("unexplained change"));
        if inf.accept {
            let mv = inf.best.expect("accepted inference has a move");
            self.lost = false;
            self.vision.commit_reference(stats);
            self.settled_at = Some(frame.at);
            self.commit(mv, "vision", Some(inf.margin), Some(frame.at));
            return;
        }
        // Promotions: the squares are unambiguous but the piece isn't visible in
        // the diff. Without an LLM, assume a queen (undo/enter a move corrects it).
        let promo = inf.best.as_ref().is_some_and(|m| m.promotion().is_some());
        let squares_clear = inf.margin >= self.vision.min_margin() && inf.residual <= self.vision.max_residual();
        if promo && squares_clear && inf.reason == Some("ambiguous") && self.llm.is_none() {
            let mv = inf.best.expect("checked above");
            self.vision.commit_reference(stats);
            self.settled_at = Some(frame.at);
            self.bus.emit(EventKind::Info { message: "promotion read as a queen; undo if it was another piece".into() });
            self.commit(mv, "vision", Some(inf.margin), Some(frame.at));
            return;
        }
        if inf.reason == Some("ambiguous") {
            // Keep the reference: the next settled read is also tried over two
            // plies against it, in case this was a real move nobody resolves.
            self.vision.note_missed(frame.at);
        }
        match inf.reason {
            Some("ambiguous") if self.llm.is_some() && !self.llm_busy && !self.over_budget() => {
                let pos = self.game.position();
                let cands: Vec<String> = if promo {
                    // Every promotion piece on the same from/to squares.
                    let best = inf.best.as_ref().expect("promo implies best");
                    pos.legal_moves()
                        .into_iter()
                        .filter(|m| m.from() == best.from() && m.to() == best.to())
                        .map(|m| game::san(pos, m))
                        .collect()
                } else {
                    inf.ranked
                        .iter()
                        .filter_map(|(u, _)| game::parse_move(pos, u))
                        .map(|m| game::san(pos, m))
                        .collect()
                };
                self.ask_pick(frame, cands);
            }
            Some("unexplained change") if lost_run => self.set_lost("the board doesn't match the game"),
            _ => {}
        }
    }

    fn set_lost(&mut self, reason: &str) {
        if !self.lost {
            self.lost = true;
            self.bus.emit(EventKind::LostTrack { reason: reason.into() });
            self.speaker.say("Lost track of the board");
            self.save_frame("lost");
        }
        self.dirty = true;
    }

    // ---------------------------------------------------------------- moves

    fn commit(&mut self, mv: shakmaty::Move, source: &str, margin: Option<f32>, settled: Option<Instant>) {
        let suggested = self.suggestion.as_ref().map(|s| s.uci.clone());
        let ply = match self.game.play(mv) {
            Ok(p) => p.clone(),
            Err(e) => {
                self.bus.emit(EventKind::Error { source: "game".into(), message: e });
                return;
            }
        };
        self.epoch += 1;
        if self.engine_inflight {
            // That search is for the old position; cut it short (queued before any ponder).
            tokio::spawn(self.engine.stop());
        }
        if let Some(t) = settled {
            self.latency.detect_ms = Some(ms(t.elapsed()));
        }
        let was_mine = ply.by == Side::You;
        self.bus.emit(EventKind::Commit {
            san: ply.san.clone(),
            uci: ply.uci.clone(),
            by: ply.by,
            suggested: if was_mine { suggested.clone() } else { None },
            margin,
            source: source.into(),
        });
        self.suggestion = None;
        self.save_pgn();
        self.save_frame("commit");

        if let Some(over) = self.game.outcome() {
            self.bus.emit(EventKind::GameOver { result: over.result.clone(), reason: over.reason.clone() });
            let line = if over.reason == "checkmate" { "Checkmate".to_string() } else { format!("Draw by {}", over.reason) };
            self.speaker.say(&line);
            self.dirty = true;
            return;
        }

        if was_mine {
            // Ponder on the reply the engine expected, if we played its move.
            let expected = self.last_reply.as_ref().filter(|r| r.bestmove == ply.uci);
            if let (true, Some(ponder)) = (self.cfg.ponder, expected.and_then(|r| r.ponder.clone())) {
                let mut pos = self.engine_pos();
                pos.moves.push(ponder);
                self.engine.ponder(pos);
            }
        } else {
            self.maybe_request_suggestion();
        }
        self.dirty = true;
    }

    fn engine_pos(&self) -> EnginePos {
        EnginePos { start_fen: self.game.start_fen().map(String::from), moves: self.game.uci_moves() }
    }

    fn maybe_request_suggestion(&mut self) {
        if !self.game.my_turn() || self.engine_inflight || self.game.outcome().is_some() {
            return;
        }
        self.engine_inflight = true;
        // Queued now, so it reaches the engine after any stop/ponder sent before it.
        let reply = self.engine.best_move(self.engine_pos());
        let asked_at = Instant::now();
        let epoch = self.epoch;
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result = reply.await;
            let _ = tx.send(Msg::Engine { epoch, result, asked_at });
        });
    }

    fn on_engine(&mut self, epoch: u64, result: Result<EngineReply, EngineError>, asked_at: Instant) {
        self.engine_inflight = false;
        if epoch != self.epoch {
            // The game moved on while the engine was thinking; ask again for the new position.
            self.maybe_request_suggestion();
            return;
        }
        match result {
            Ok(reply) => {
                self.engine_failed_at = None;
                self.engine_error_spoken = false;
                let pos = self.game.position();
                let Some(mv) = game::parse_move(pos, &reply.bestmove) else {
                    self.bus.emit(EventKind::Error { source: "engine".into(), message: format!("bad move {}", reply.bestmove) });
                    return;
                };
                let san = game::san(pos, mv);
                let spoken = spoken_san(&san);
                let engine_ms = ms(asked_at.elapsed());
                self.speaker.say(&spoken);
                self.latency.engine_ms = Some(engine_ms);
                self.latency.ponder_hit = Some(reply.ponder_hit);
                self.latency.total_ms = self.settled_at.take().map(|t| ms(t.elapsed()));
                self.bus.emit(EventKind::Suggest {
                    san: san.clone(),
                    uci: reply.bestmove.clone(),
                    spoken: spoken.clone(),
                    score_text: reply.score.text(),
                    ponder_hit: reply.ponder_hit,
                    engine_ms,
                });
                self.suggestion = Some(Suggestion {
                    uci: reply.bestmove.clone(),
                    san,
                    spoken,
                    score: reply.score,
                    score_text: reply.score.text(),
                    confidence: reply.score.confidence(),
                });
                self.last_reply = Some(reply);
            }
            Err(EngineError::GameOver | EngineError::Cancelled) => {}
            Err(e) => {
                self.bus.emit(EventKind::Error { source: "engine".into(), message: e.to_string() });
                if !self.engine_error_spoken {
                    self.speaker.say("Something went wrong. Still trying.");
                    self.engine_error_spoken = true;
                }
                self.engine_failed_at = Some(Instant::now());
            }
        }
        self.dirty = true;
    }

    // ---------------------------------------------------------------- LLM

    fn over_budget(&self) -> bool {
        self.spent >= self.cfg.max_cost
    }

    fn maybe_locate(&mut self, frame: &Frame) {
        let Some(llm) = self.llm.clone() else { return };
        if self.llm_busy || self.over_budget() || self.last_locate.is_some_and(|t| t.elapsed() < LOCATE_EVERY) {
            return;
        }
        self.llm_busy = true;
        self.last_locate = Some(Instant::now());
        let tx = self.tx.clone();
        let frame = frame.clone();
        tokio::spawn(async move {
            let jpeg = encode_jpeg(&frame).await;
            let result = llm.locate_board(&jpeg).await;
            let _ = tx.send(Msg::Located { frame, result });
        });
    }

    fn ask_pick(&mut self, frame: &Frame, candidates: Vec<String>) {
        let Some(llm) = self.llm.clone() else { return };
        if candidates.is_empty() {
            return;
        }
        self.llm_busy = true;
        let tx = self.tx.clone();
        let frame = frame.clone();
        let fen = self.game.fen();
        let epoch = self.epoch;
        tokio::spawn(async move {
            let jpeg = encode_jpeg(&frame).await;
            let result = llm.pick_move(&jpeg, &fen, &candidates).await;
            let _ = tx.send(Msg::Picked { epoch, frame_at: frame.at, result });
        });
    }

    fn charge(&mut self, cost: Cost, purpose: &str) {
        if cost.usd <= 0.0 {
            return;
        }
        self.spent += cost.usd;
        self.bus.emit(EventKind::Spend { usd: cost.usd, total_usd: self.spent, purpose: purpose.into() });
        if self.over_budget() && !self.paused {
            self.paused = true;
            self.bus.emit(EventKind::Paused { reason: "spending limit reached".into() });
            if !self.budget_spoken {
                self.speaker.say("Spending limit reached. Paused.");
                self.budget_spoken = true;
            }
        }
        self.dirty = true;
    }

    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Engine { epoch, result, asked_at } => self.on_engine(epoch, result, asked_at),
            Msg::Located { frame, result } => {
                self.llm_busy = false;
                match result {
                    Ok((corners, cost)) => {
                        self.charge(cost, "locate board");
                        if let (Some(view), None) = (corners, self.vision.corners()) {
                            self.adopt_view_corners(&frame, view, "llm");
                        }
                    }
                    Err(e) => {
                        self.bus.emit(EventKind::Error { source: "llm".into(), message: e.to_string() });
                    }
                }
            }
            Msg::Picked { epoch, frame_at, result } => {
                self.llm_busy = false;
                match result {
                    Ok((answer, cost)) => {
                        self.charge(cost, "pick move");
                        if epoch != self.epoch {
                            return;
                        }
                        if let Some(mv) = answer.and_then(|san| self.game.parse_move(&san)) {
                            // The reference must be re-taken: our stats were for an ambiguous read.
                            self.vision.rebaseline();
                            self.commit(mv, "llm", None, Some(frame_at));
                        }
                    }
                    Err(e) => {
                        self.bus.emit(EventKind::Error { source: "llm".into(), message: e.to_string() });
                    }
                }
            }
        }
        self.dirty = true;
    }

    /// `view` corners are TL, TR, BR, BL fractions of the frame.
    fn adopt_view_corners(&mut self, frame: &Frame, view: [[f32; 2]; 4], source: &str) {
        let (w, h) = (frame.width as f32, frame.height as f32);
        let px = view.map(|[x, y]| [x * w, y * h]);
        let corners = Corners::from_view(px, self.cfg.my_color == Color::White);
        self.vision.set_corners(frame, corners);
        self.lost = false;
        self.unexplained.reset();
        self.bus.emit(EventKind::BoardFound { source: source.into() });
        self.dirty = true;
    }

    // ---------------------------------------------------------------- controls

    fn on_control(&mut self, c: Control) {
        match c {
            Control::Pause => {
                self.paused = true;
                self.bus.emit(EventKind::Paused { reason: "requested".into() });
            }
            Control::Resume => {
                self.paused = false;
                self.budget_spoken = false;
                self.vision.rebaseline();
                self.bus.emit(EventKind::Resumed);
                self.maybe_request_suggestion();
            }
            Control::Undo => {
                if let Some(ply) = self.game.undo() {
                    self.epoch += 1;
                    self.suggestion = None;
                    self.last_reply = None;
                    self.vision.rebaseline();
                    self.lost = false;
                    self.bus.emit(EventKind::Undo { san: ply.san });
                    self.save_pgn();
                    tokio::spawn(self.engine.stop());
                    self.maybe_request_suggestion();
                }
            }
            Control::Move { mv } => match self.game.parse_move(&mv) {
                Some(m) => {
                    self.vision.rebaseline();
                    self.lost = false;
                    self.commit(m, "manual", None, None);
                }
                None => {
                    self.bus.emit(EventKind::Error { source: "control".into(), message: format!("illegal move {mv}") });
                }
            },
            Control::NewGame { color, fen } => {
                let my_color = if color == "black" { Color::Black } else { Color::White };
                match GameState::new(my_color, fen.as_deref()) {
                    Ok(g) => {
                        self.cfg.my_color = my_color;
                        self.cfg.fen = fen;
                        self.game = g;
                        self.epoch += 1;
                        self.suggestion = None;
                        self.last_reply = None;
                        self.lost = false;
                        self.vision.rebaseline();
                        tokio::spawn(self.engine.new_game());
                        self.bus.emit(EventKind::NewGame { color: color_name(my_color).into(), fen: self.game.fen() });
                        self.save_pgn();
                        self.maybe_request_suggestion();
                    }
                    Err(e) => {
                        self.bus.emit(EventKind::Error { source: "control".into(), message: e });
                    }
                }
            }
            Control::Repeat => {
                if let Some(s) = &self.suggestion {
                    self.speaker.say(&s.spoken);
                }
            }
            Control::SetCorners { view } => match self.latest_frame.clone() {
                Some(f) => self.adopt_view_corners(&f, view, "manual"),
                None => {
                    self.bus.emit(EventKind::Error { source: "control".into(), message: "no frame yet".into() });
                }
            },
            Control::Relocate => {
                self.vision.forget_board();
                self.last_locate = None;
            }
        }
        self.dirty = true;
    }

    fn on_capture(&mut self, s: CaptureStatus) {
        match s {
            CaptureStatus::Running => self.source_missing = false,
            CaptureStatus::Missing(message) => {
                self.source_missing = true;
                self.bus.emit(EventKind::SourceMissing { message });
            }
            CaptureStatus::Error(message) => {
                self.bus.emit(EventKind::Error { source: "capture".into(), message });
            }
            CaptureStatus::Ended => {
                self.ended = true;
                self.bus.emit(EventKind::Info { message: "recording finished".into() });
            }
        }
        self.dirty = true;
    }

    fn on_tick(&mut self) {
        // Last resort for a missed read that neither two-ply inference nor the
        // LLM resolved: ask the user, once, to enter the move.
        match self.vision.missed_since() {
            None => self.missed_spoken = false,
            Some(t) if !self.missed_spoken && !self.paused && !self.llm_busy && t.elapsed() >= MISSED_ANNOUNCE => {
                self.missed_spoken = true;
                self.bus.emit(EventKind::Info { message: "didn't catch the last move; enter it to continue".into() });
                self.speaker.say("Didn't catch that move");
                self.save_frame("missed");
                self.dirty = true;
            }
            Some(_) => {}
        }
        if self.engine_failed_at.is_some_and(|t| t.elapsed() >= ENGINE_RETRY) && !self.paused {
            self.engine_failed_at = None;
            self.maybe_request_suggestion();
        }
        // Frame counters and per-frame cost refresh at the tick rate, not per frame.
        if self.latest_frame.is_some() {
            self.latency.frame_ms = Some(self.vision.last_frame_ms);
            self.dirty = true;
        }
    }

    // ---------------------------------------------------------------- output

    fn status(&self) -> Status {
        if self.game.outcome().is_some() {
            Status::GameOver
        } else if self.ended {
            Status::RecordingFinished
        } else if self.paused {
            Status::Paused
        } else if self.source_missing {
            Status::CantSeeCall
        } else if self.lost {
            Status::LostTrack
        } else if self.vision.corners().is_none() {
            Status::LookingForBoard
        } else if self.game.my_turn() {
            Status::YourMove
        } else {
            Status::WaitingForOpponent
        }
    }

    fn publish(&mut self) {
        self.dirty = false;
        let status = self.status();
        let f = self.latest_frame.as_ref();
        self.bus.publish(Snapshot {
            status,
            status_label: status.label().into(),
            paused: self.paused,
            my_color: color_name(self.cfg.my_color).into(),
            turn: color_name(self.game.position().turn()).into(),
            fen: self.game.fen(),
            moves: self.game.plies().iter().map(|p| p.san.clone()).collect(),
            suggestion: self.suggestion.clone().filter(|_| self.game.my_turn()),
            last_frame_seq: f.map_or(0, |f| f.seq),
            last_frame_ts_ms: f.map_or(0, |f| f.ts_ms),
            frame_size: f.map(|f| [f.width, f.height]),
            corners: self.vision.corners(),
            spent_usd: self.spent,
            max_cost_usd: self.cfg.max_cost,
            model: self.llm.as_ref().map(|l| l.model().to_string()),
            llm_enabled: self.llm.is_some(),
            latency: self.latency,
        });
    }

    fn save_pgn(&self) {
        let Some(dir) = &self.cfg.run_dir else { return };
        let pgn = self.game.pgn();
        let dir = dir.clone();
        tokio::task::spawn_blocking(move || {
            let tmp = dir.join("game.pgn.tmp");
            let _ = std::fs::write(&tmp, pgn).and_then(|_| std::fs::rename(&tmp, dir.join("game.pgn")));
        });
    }

    fn save_frame(&self, what: &str) {
        let (Some(dir), Some(frame)) = (&self.cfg.run_dir, self.latest_frame.clone()) else { return };
        let path = dir.join("frames").join(format!("{}-{what}.jpg", frame.ts_ms));
        tokio::spawn(async move {
            let jpeg = encode_jpeg(&frame).await;
            let _ = tokio::fs::write(path, jpeg).await;
        });
    }
}

/// Run of settled reads in a row that no legal move explains.
#[derive(Default)]
struct UnexplainedRun(u32);

impl UnexplainedRun {
    /// Record one settled read. Returns true once `UNEXPLAINED_LIMIT`
    /// unexplained reads have come in a row.
    fn record(&mut self, unexplained: bool) -> bool {
        self.0 = if unexplained { self.0 + 1 } else { 0 };
        self.0 >= UNEXPLAINED_LIMIT
    }

    fn reset(&mut self) {
        self.0 = 0;
    }
}

pub async fn encode_jpeg(frame: &Frame) -> Vec<u8> {
    let frame = frame.clone();
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let enc = jpeg_encoder::Encoder::new(&mut out, 80);
        let _ = enc.encode(&frame.rgb, frame.width as u16, frame.height as u16, jpeg_encoder::ColorType::Rgb);
        out
    })
    .await
    .unwrap_or_default()
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Rig;
    use shakmaty::{Chess, Square};

    /// Reason of the settled read of the start position, with an optional hand on the board.
    fn read(rig: &mut Rig, hand: Option<(Square, f32)>) -> Option<&'static str> {
        let pos = Chess::default();
        match rig.settle(Some(pos.board()), hand, &pos) {
            Some(Outcome::Read { inference, .. }) => inference.reason,
            _ => panic!("expected a fresh read"),
        }
    }

    #[test]
    fn unexplained_counts_only_consecutive_reads() {
        let pos = Chess::default();
        let mut rig = Rig::new(pos.board());
        assert!(matches!(rig.settle(Some(pos.board()), None, &pos), Some(Outcome::Baseline)));
        let mut run = UnexplainedRun::default();

        // Sporadic misreads (a hand resting on the board, then gone): each is
        // followed by a clean "no change" read, so the run never builds up.
        for sq in [Square::E5, Square::C5, Square::D5, Square::B5, Square::G5] {
            let reason = read(&mut rig, Some((sq, 45.0)));
            assert_eq!(reason, Some("unexplained change"));
            assert!(!run.record(reason == Some("unexplained change")), "lost on a sporadic misread");
            let reason = read(&mut rig, None);
            assert_eq!(reason, Some("no change"));
            assert!(!run.record(reason == Some("unexplained change")));
        }

        // Three unexplained reads in a row do declare the board lost.
        let hits: Vec<bool> = [Square::D5, Square::B5, Square::G5]
            .into_iter()
            .map(|sq| {
                let reason = read(&mut rig, Some((sq, 45.0)));
                assert_eq!(reason, Some("unexplained change"));
                run.record(true)
            })
            .collect();
        assert_eq!(hits, [false, false, true]);
    }
}
