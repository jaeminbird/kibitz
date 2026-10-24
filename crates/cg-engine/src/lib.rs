//! Persistent Stockfish (UCI) engine with pondering, plus locating and
//! downloading a pinned build.
//!
//! The engine runs as one actor task that owns the child process. A process
//! that times out or crashes is killed and never reused.

mod actor;
mod install;
mod uci;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use cg_core::Score;
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Outcome, Position as _};
use tokio::sync::{mpsc, oneshot};

pub use install::{data_dir, download, locate};

/// Engine settings. Build with [`EngineConfig::new`] and override fields as needed.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Path to the UCI engine binary.
    pub path: PathBuf,
    /// Search time per move (`go movetime`).
    pub movetime: Duration,
    /// Engine threads (`Threads`).
    pub threads: u32,
    /// Hash table size in MiB (`Hash`).
    pub hash_mb: u32,
    /// How long the start-up handshake may take.
    pub handshake_timeout: Duration,
    /// Extra time beyond `movetime` before a search counts as timed out.
    pub slack: Duration,
}

impl EngineConfig {
    /// Defaults: 300 ms movetime, one fewer thread than CPUs (min 1), 128 MiB hash,
    /// 10 s handshake timeout, 2 s slack.
    pub fn new(path: PathBuf) -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        EngineConfig {
            path,
            movetime: Duration::from_millis(300),
            threads: (cpus.saturating_sub(1)).max(1) as u32,
            hash_mb: 128,
            handshake_timeout: Duration::from_secs(10),
            slack: Duration::from_secs(2),
        }
    }
}

/// A position as the starting point plus the UCI moves played from it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Position {
    /// FEN to start from; `None` means the standard starting position.
    pub start_fen: Option<String>,
    /// Moves in UCI notation, in order.
    pub moves: Vec<String>,
}

/// The engine's answer to a [`Engine::best_move`] request.
#[derive(Clone, Debug)]
pub struct EngineReply {
    /// Best move in UCI notation.
    pub bestmove: String,
    /// Expected reply in UCI notation, if the engine gave one.
    pub ponder: Option<String>,
    /// Evaluation from the side-to-move point of view (last `multipv 1` info line).
    /// `Cp(0)` if the engine sent no score.
    pub score: Score,
    /// Depth of the last `multipv 1` info line; 0 if none.
    pub depth: u32,
    /// Wall time of the whole request, including any restart.
    pub elapsed: Duration,
    /// True if the answer came from a ponder search that matched the position.
    pub ponder_hit: bool,
}

/// Errors from the engine and its setup.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("stockfish is not installed; run `kibitz setup`")]
    NotInstalled,
    #[error("the game is over")]
    GameOver,
    #[error("engine timed out")]
    Timeout,
    #[error("engine crashed: {0}")]
    Crashed(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("invalid position: {0}")]
    BadPosition(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Requests sent from [`Engine`] handles to the actor task.
pub(crate) enum Request {
    BestMove {
        pos: Position,
        reply: oneshot::Sender<Result<EngineReply, EngineError>>,
    },
    Ponder {
        pos: Position,
    },
    Stop {
        ack: oneshot::Sender<()>,
    },
    NewGame {
        ack: oneshot::Sender<()>,
    },
    Shutdown {
        ack: oneshot::Sender<()>,
    },
}

/// Cheap handle to the engine actor. Clones share one engine process.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::UnboundedSender<Request>,
    name: Arc<str>,
}

impl Engine {
    /// Spawn the process and complete the UCI handshake.
    pub async fn start(cfg: EngineConfig) -> Result<Engine, EngineError> {
        let mut actor = actor::Actor::new(cfg);
        actor.connect().await?;
        let name: Arc<str> = actor.name().into();
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(actor::run(actor, rx));
        Ok(Engine { tx, name })
    }

    /// Best move for `pos`.
    ///
    /// The position is checked with shakmaty first: an illegal move gives
    /// `BadPosition`, and a finished game gives `GameOver` without touching the engine.
    /// On `Timeout` or `Crashed` the process is killed, a fresh one is started, and the
    /// request is retried once. A second failure is returned.
    pub async fn best_move(&self, pos: Position) -> Result<EngineReply, EngineError> {
        check_playable(&pos)?;
        let (reply, rx) = oneshot::channel();
        self.send(Request::BestMove { pos, reply })?;
        rx.await.map_err(|_| actor_gone())?
    }

    /// Start pondering `pos` (normally the position after our suggested move and its
    /// expected reply). Returns without waiting. Replaces any previous ponder.
    /// Invalid or finished positions are ignored with a warning.
    pub async fn ponder(&self, pos: Position) {
        if let Err(e) = check_playable(&pos) {
            tracing::warn!(error = %e, "not pondering");
            return;
        }
        if self.tx.send(Request::Ponder { pos }).is_err() {
            tracing::warn!("engine actor is gone; not pondering");
        }
    }

    /// Stop any ponder search and discard its result. Used on undo or new game.
    pub async fn stop(&self) {
        let (ack, rx) = oneshot::channel();
        if self.send(Request::Stop { ack }).is_ok() {
            let _ = rx.await;
        }
    }

    /// `ucinewgame` followed by `isready`, which clears the hash table.
    pub async fn new_game(&self) {
        let (ack, rx) = oneshot::channel();
        if self.send(Request::NewGame { ack }).is_ok() {
            let _ = rx.await;
        }
    }

    /// Engine id name from the handshake, e.g. "Stockfish 19".
    pub fn name(&self) -> String {
        self.name.to_string()
    }

    /// Stop the engine process and wait for it to exit.
    pub async fn shutdown(self) {
        let (ack, rx) = oneshot::channel();
        if self.send(Request::Shutdown { ack }).is_ok() {
            let _ = rx.await;
        }
    }

    fn send(&self, req: Request) -> Result<(), EngineError> {
        self.tx.send(req).map_err(|_| actor_gone())
    }
}

fn actor_gone() -> EngineError {
    EngineError::Crashed("engine actor stopped".into())
}

/// Replays `pos` with shakmaty. Returns `BadPosition` for an unparsable or illegal
/// position, and `GameOver` if the position has a known outcome.
pub(crate) fn check_playable(pos: &Position) -> Result<Chess, EngineError> {
    let mut chess = match &pos.start_fen {
        None => Chess::default(),
        Some(fen) => Fen::from_ascii(fen.as_bytes())
            .map_err(|e| EngineError::BadPosition(e.to_string()))?
            .into_position::<Chess>(CastlingMode::Standard)
            .map_err(|e| EngineError::BadPosition(e.to_string()))?,
    };
    for uci in &pos.moves {
        let mv = uci
            .parse::<UciMove>()
            .map_err(|e| EngineError::BadPosition(format!("{uci}: {e}")))?
            .to_move(&chess)
            .map_err(|e| EngineError::BadPosition(format!("{uci}: {e}")))?;
        chess = chess
            .play(mv)
            .map_err(|e| EngineError::BadPosition(format!("{uci}: {e}")))?;
    }
    if !matches!(chess.outcome(), Outcome::Unknown) {
        return Err(EngineError::GameOver);
    }
    Ok(chess)
}
