use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    You,
    Opponent,
}

/// A command from the command center. Validated by the server, queued, and
/// applied by the game loop between frames.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Control {
    Pause,
    Resume,
    /// Take back the last move.
    Undo,
    /// Enter a move the camera missed. SAN or UCI.
    Move { mv: String },
    /// Start over. `color` is "white" or "black"; `fen` optional starting position.
    NewGame { color: String, fen: Option<String> },
    /// Say the current suggestion again.
    Repeat,
    /// Manually set the board corners, in the viewer's on-screen order
    /// TL, TR, BR, BL, as fractions (0..1) of the frame width/height.
    SetCorners { view: [[f32; 2]; 4] },
    /// Forget the board location and find it again.
    Relocate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_shape() {
        let c: Control = serde_json::from_str(r#"{"action":"move","mv":"e4"}"#).unwrap();
        assert_eq!(c, Control::Move { mv: "e4".into() });
        let c: Control = serde_json::from_str(r#"{"action":"new_game","color":"black"}"#).unwrap();
        assert_eq!(c, Control::NewGame { color: "black".into(), fen: None });
        let c: Control = serde_json::from_str(r#"{"action":"pause"}"#).unwrap();
        assert_eq!(c, Control::Pause);
    }
}
