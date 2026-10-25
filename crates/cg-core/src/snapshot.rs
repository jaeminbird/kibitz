use serde::{Deserialize, Serialize};

/// What the command center status pill shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    YourMove,
    WaitingForOpponent,
    Paused,
    LookingForBoard,
    LostTrack,
    CantSeeCall,
    GameOver,
    RecordingFinished,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::YourMove => "Your move",
            Status::WaitingForOpponent => "Waiting for opponent",
            Status::Paused => "Paused",
            Status::LookingForBoard => "Looking for the board",
            Status::LostTrack => "Lost track",
            Status::CantSeeCall => "Can't see the call",
            Status::GameOver => "Game over",
            Status::RecordingFinished => "Recording finished",
        }
    }
}

/// Outer corners of the playing area in image pixel coordinates, in BOARD
/// order: `[a8, h8, h1, a1]` (i.e. top-left, top-right, bottom-right,
/// bottom-left when viewed from White's side).
///
/// Use [`Corners::from_view`] to convert corners given in the viewer's
/// on-screen order (TL, TR, BR, BL) for a player of a given color.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Corners(pub [[f32; 2]; 4]);

impl Corners {
    /// `view` is TL, TR, BR, BL as seen on screen from the player's seat
    /// (the player's own pieces are at the bottom of the picture).
    pub fn from_view(view: [[f32; 2]; 4], player_is_white: bool) -> Corners {
        if player_is_white {
            Corners(view)
        } else {
            // From Black's seat, screen TL is h1, TR is a1, BR is a8, BL is h8.
            let [tl, tr, br, bl] = view;
            Corners([br, bl, tl, tr])
        }
    }

    /// Inverse of [`Corners::from_view`].
    pub fn to_view(self, player_is_white: bool) -> [[f32; 2]; 4] {
        if player_is_white {
            self.0
        } else {
            let [a8, h8, h1, a1] = self.0;
            [h1, a1, a8, h8]
        }
    }
}

/// Why [`validate_corners`] rejected a quadrilateral.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadError {
    /// A coordinate is not finite or lies outside `[-margin, 1 + margin]`.
    OutOfRange,
    /// The quadrilateral covers 1% of the frame or less.
    Degenerate,
}

/// Checks four corners given as fractions of frame width and height: every
/// coordinate is finite and within `[-margin, 1 + margin]`, and the
/// quadrilateral spans more than 1% of the frame (shoelace area).
pub fn validate_corners(quad: &[[f64; 2]; 4], margin: f64) -> Result<(), QuadError> {
    if !quad
        .iter()
        .flatten()
        .all(|v| (-margin..=1.0 + margin).contains(v))
    {
        return Err(QuadError::OutOfRange);
    }
    let twice_area: f64 = (0..4)
        .map(|i| {
            let (p, q) = (quad[i], quad[(i + 1) % 4]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum();
    if twice_area.abs() / 2.0 <= 0.01 {
        return Err(QuadError::Degenerate);
    }
    Ok(())
}

/// Engine evaluation from the asking player's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Score {
    /// Centipawns; positive is good for the player.
    Cp(i32),
    /// Mate in N moves; positive means the player mates, negative means gets mated.
    Mate(i32),
}

impl Score {
    /// "+1.5", "-0.3", "mate in 3", "mated in 2".
    pub fn text(self) -> String {
        match self {
            Score::Cp(cp) => format!("{:+.1}", cp as f32 / 100.0),
            Score::Mate(n) if n > 0 => format!("mate in {n}"),
            Score::Mate(n) => format!("mated in {}", -n),
        }
    }

    /// 0..1 value for the confidence bar (logistic on centipawns).
    pub fn confidence(self) -> f32 {
        match self {
            Score::Cp(cp) => 1.0 / (1.0 + (-(cp as f32) / 250.0).exp()),
            Score::Mate(n) if n > 0 => 1.0,
            Score::Mate(_) => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    pub uci: String,
    pub san: String,
    /// e.g. "Rook to d1"
    pub spoken: String,
    pub score: Score,
    pub score_text: String,
    pub confidence: f32,
}

/// Timings of the most recent detect → suggest cycle, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    /// Board settled (stable frame) → move committed.
    pub detect_ms: Option<f32>,
    /// Move committed → engine replied.
    pub engine_ms: Option<f32>,
    /// Board settled → speech started.
    pub total_ms: Option<f32>,
    /// Per-frame vision cost of the last processed frame.
    pub frame_ms: Option<f32>,
    /// Whether the last engine answer came from a ponder hit.
    pub ponder_hit: Option<bool>,
}

/// The game's current picture. Rebuilt after every change, broadcast, never logged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub status: Status,
    pub status_label: String,
    pub paused: bool,
    /// "white" | "black"
    pub my_color: String,
    /// "white" | "black"
    pub turn: String,
    pub fen: String,
    /// SAN, in order.
    pub moves: Vec<String>,
    pub suggestion: Option<Suggestion>,
    pub last_frame_seq: u64,
    pub last_frame_ts_ms: u64,
    pub frame_size: Option<[u32; 2]>,
    pub corners: Option<Corners>,
    pub spent_usd: f64,
    pub max_cost_usd: f64,
    /// Vision LLM model name if the fallback is enabled.
    pub model: Option<String>,
    pub llm_enabled: bool,
    pub latency: Latency,
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot {
            status: Status::LookingForBoard,
            status_label: Status::LookingForBoard.label().to_string(),
            paused: false,
            my_color: "white".into(),
            turn: "white".into(),
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into(),
            moves: vec![],
            suggestion: None,
            last_frame_seq: 0,
            last_frame_ts_ms: 0,
            frame_size: None,
            corners: None,
            spent_usd: 0.0,
            max_cost_usd: 10.0,
            model: None,
            llm_enabled: false,
            latency: Latency::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_roundtrip() {
        let v = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for white in [true, false] {
            assert_eq!(Corners::from_view(v, white).to_view(white), v);
        }
        // Black's screen bottom-right is a8.
        assert_eq!(Corners::from_view(v, false).0[0], [1.0, 1.0]);
    }

    #[test]
    fn corners_validation() {
        let square = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
        assert_eq!(validate_corners(&square, 0.05), Ok(()));
        let collinear = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.0]];
        assert_eq!(
            validate_corners(&collinear, 0.05),
            Err(QuadError::Degenerate)
        );
        let outside = [[0.1, 0.1], [1.08, 0.1], [0.9, 0.9], [0.1, 0.9]];
        assert_eq!(validate_corners(&outside, 0.05), Err(QuadError::OutOfRange));
        assert_eq!(validate_corners(&outside, 0.1), Ok(()));
        let nan = [[f64::NAN, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
        assert_eq!(validate_corners(&nan, 0.1), Err(QuadError::OutOfRange));
    }

    #[test]
    fn score_text() {
        assert_eq!(Score::Cp(150).text(), "+1.5");
        assert_eq!(Score::Mate(3).text(), "mate in 3");
        assert_eq!(Score::Mate(-2).text(), "mated in 2");
        assert!((Score::Cp(0).confidence() - 0.5).abs() < 1e-6);
    }
}
