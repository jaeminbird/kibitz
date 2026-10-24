use serde::{Deserialize, Serialize};

use crate::control::Side;

/// One thing that happened in a game.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GameEvent {
    /// Unix milliseconds.
    pub ts_ms: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// A move was accepted into the game.
    Commit {
        san: String,
        uci: String,
        by: Side,
        /// The suggestion that was active when the move was made (your moves only).
        suggested: Option<String>,
        /// How clearly this move beat the alternatives (vision), or None if entered manually.
        margin: Option<f32>,
        /// "vision" | "llm" | "manual"
        source: String,
    },
    /// The engine suggested a move.
    Suggest {
        san: String,
        uci: String,
        spoken: String,
        score_text: String,
        ponder_hit: bool,
        engine_ms: f32,
    },
    /// A frame was read. Only emitted for frames that reached move inference
    /// (stable frames with a change), not for every captured frame.
    Frame {
        seq: u64,
        changed_squares: u32,
        best: Option<String>,
        margin: Option<f32>,
        /// Why no move was committed, if none was.
        skipped: Option<String>,
        vision_ms: f32,
    },
    BoardFound { source: String },
    LostTrack { reason: String },
    SourceMissing { message: String },
    Paused { reason: String },
    Resumed,
    NewGame { color: String, fen: String },
    Undo { san: String },
    GameOver { result: String, reason: String },
    Spend { usd: f64, total_usd: f64, purpose: String },
    Error { source: String, message: String },
    Info { message: String },
}

impl EventKind {
    /// True for the "problems" log filter.
    pub fn is_problem(&self) -> bool {
        matches!(
            self,
            EventKind::LostTrack { .. } | EventKind::SourceMissing { .. } | EventKind::Error { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattened_json() {
        let e = GameEvent { ts_ms: 5, kind: EventKind::Resumed };
        assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"ts_ms":5,"kind":"resumed"}"#);
    }
}
