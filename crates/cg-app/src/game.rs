//! The game record: start position, moves played, whose side made them.
//! Pure and synchronous; the orchestrator owns exactly one of these.

use cg_core::Side;
use shakmaty::fen::{Epd, Fen};
use shakmaty::san::{San, SanPlus};
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, KnownOutcome, Move, Position};

#[derive(Clone, Debug)]
pub struct Ply {
    pub san: String,
    pub uci: String,
    pub by: Side,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub my_color: Color,
    start: Chess,
    start_fen: Option<String>,
    plies: Vec<Ply>,
    /// positions[i] is the position before plies[i]; positions.last() is current.
    positions: Vec<Chess>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameOver {
    /// "1-0", "0-1", "1/2-1/2"
    pub result: String,
    pub reason: String,
}

impl GameState {
    pub fn new(my_color: Color, fen: Option<&str>) -> Result<GameState, String> {
        let start: Chess = match fen {
            Some(f) => Fen::from_ascii(f.trim().as_bytes())
                .map_err(|e| format!("bad FEN: {e}"))?
                .into_position(CastlingMode::Standard)
                .map_err(|e| format!("illegal position: {e}"))?,
            None => Chess::default(),
        };
        Ok(GameState {
            my_color,
            start_fen: fen.map(|f| f.trim().to_string()),
            positions: vec![start.clone()],
            start,
            plies: vec![],
        })
    }

    pub fn position(&self) -> &Chess {
        self.positions.last().expect("at least the start position")
    }

    pub fn plies(&self) -> &[Ply] {
        &self.plies
    }

    pub fn my_turn(&self) -> bool {
        self.position().turn() == self.my_color
    }

    pub fn side_to_move(&self) -> Side {
        if self.my_turn() {
            Side::You
        } else {
            Side::Opponent
        }
    }

    pub fn fen(&self) -> String {
        Fen::from_position(self.position(), EnPassantMode::Legal).to_string()
    }

    pub fn start_fen(&self) -> Option<&str> {
        self.start_fen.as_deref()
    }

    /// UCI moves from the start, for the engine's `position … moves …`.
    pub fn uci_moves(&self) -> Vec<String> {
        self.plies.iter().map(|p| p.uci.clone()).collect()
    }

    /// Parse SAN ("Nf3", "O-O", "exd5+") or UCI ("g1f3") against the current position.
    pub fn parse_move(&self, text: &str) -> Option<Move> {
        parse_move(self.position(), text)
    }

    /// Play a legal move. Returns the recorded ply.
    pub fn play(&mut self, mv: Move) -> Result<&Ply, String> {
        let pos = self.position().clone();
        if !pos.is_legal(mv) {
            return Err(format!("illegal move {}", uci(&mv)));
        }
        let san = SanPlus::from_move(pos.clone(), mv).to_string();
        let by = self.side_to_move();
        let mut next = pos;
        next.play_unchecked(mv);
        self.plies.push(Ply { san, uci: uci(&mv), by });
        self.positions.push(next);
        Ok(self.plies.last().unwrap())
    }

    /// Take back the last move.
    pub fn undo(&mut self) -> Option<Ply> {
        let ply = self.plies.pop()?;
        self.positions.pop();
        Some(ply)
    }

    pub fn outcome(&self) -> Option<GameOver> {
        let pos = self.position();
        if let Some(known) = pos.outcome().known() {
            let reason = if pos.is_checkmate() {
                "checkmate"
            } else if pos.is_stalemate() {
                "stalemate"
            } else {
                "insufficient material"
            };
            return Some(GameOver { result: result_text(known), reason: reason.into() });
        }
        if pos.halfmoves() >= 100 {
            return Some(GameOver { result: "1/2-1/2".into(), reason: "fifty-move rule".into() });
        }
        let key = |p: &Chess| Epd::from_position(p, EnPassantMode::Legal).to_string();
        let now = key(pos);
        if self.positions.iter().filter(|p| key(p) == now).count() >= 3 {
            return Some(GameOver { result: "1/2-1/2".into(), reason: "threefold repetition".into() });
        }
        None
    }

    /// The game so far as PGN, "You" vs "Opponent".
    pub fn pgn(&self) -> String {
        let (white, black) = match self.my_color {
            Color::White => ("You", "Opponent"),
            Color::Black => ("Opponent", "You"),
        };
        let result = self.outcome().map(|o| o.result).unwrap_or_else(|| "*".into());
        let mut out = format!("[Event \"chessglass\"]\n[White \"{white}\"]\n[Black \"{black}\"]\n[Result \"{result}\"]\n");
        if let Some(fen) = &self.start_fen {
            out.push_str(&format!("[SetUp \"1\"]\n[FEN \"{fen}\"]\n"));
        }
        out.push('\n');
        let mut num = self.start.fullmoves().get();
        let mut white_to_move = self.start.turn() == Color::White;
        let mut tokens = Vec::new();
        for (i, ply) in self.plies.iter().enumerate() {
            if white_to_move {
                tokens.push(format!("{num}."));
            } else if i == 0 {
                tokens.push(format!("{num}..."));
            }
            tokens.push(ply.san.clone());
            if !white_to_move {
                num += 1;
            }
            white_to_move = !white_to_move;
        }
        tokens.push(result);
        out.push_str(&tokens.join(" "));
        out.push('\n');
        out
    }
}

pub fn parse_move(pos: &Chess, text: &str) -> Option<Move> {
    let t = text.trim();
    if let Ok(san) = t.parse::<SanPlus>() {
        if let Ok(m) = san.san.to_move(pos) {
            return Some(m);
        }
    }
    if let Ok(san) = t.parse::<San>() {
        if let Ok(m) = san.to_move(pos) {
            return Some(m);
        }
    }
    t.parse::<UciMove>().ok()?.to_move(pos).ok()
}

pub fn uci(mv: &Move) -> String {
    mv.to_uci(CastlingMode::Standard).to_string()
}

pub fn san(pos: &Chess, mv: Move) -> String {
    SanPlus::from_move(pos.clone(), mv).to_string()
}

fn result_text(o: KnownOutcome) -> String {
    match o {
        KnownOutcome::Decisive { winner: Color::White } => "1-0".into(),
        KnownOutcome::Decisive { winner: Color::Black } => "0-1".into(),
        KnownOutcome::Draw => "1/2-1/2".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play_all(g: &mut GameState, moves: &[&str]) {
        for m in moves {
            let mv = g.parse_move(m).unwrap_or_else(|| panic!("parse {m}"));
            g.play(mv).unwrap();
        }
    }

    #[test]
    fn sides_and_undo() {
        let mut g = GameState::new(Color::Black, None).unwrap();
        assert_eq!(g.side_to_move(), Side::Opponent);
        play_all(&mut g, &["e4", "e7e5", "Nf3"]);
        assert_eq!(g.plies()[0].by, Side::Opponent);
        assert_eq!(g.plies()[1].by, Side::You);
        assert_eq!(g.uci_moves(), ["e2e4", "e7e5", "g1f3"]);
        assert_eq!(g.undo().unwrap().san, "Nf3");
        assert_eq!(g.plies().len(), 2);
        assert!(g.my_turn() == false);
    }

    #[test]
    fn castling_uci_is_standard() {
        let mut g = GameState::new(Color::White, None).unwrap();
        play_all(&mut g, &["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]);
        assert_eq!(g.plies().last().unwrap().uci, "e1g1");
    }

    #[test]
    fn mate_and_pgn() {
        let mut g = GameState::new(Color::White, None).unwrap();
        play_all(&mut g, &["f3", "e5", "g4", "Qh4#"]);
        let o = g.outcome().unwrap();
        assert_eq!((o.result.as_str(), o.reason.as_str()), ("0-1", "checkmate"));
        let pgn = g.pgn();
        assert!(pgn.contains("[White \"You\"]"));
        assert!(pgn.trim_end().ends_with("1. f3 e5 2. g4 Qh4# 0-1"), "{pgn}");
    }

    #[test]
    fn start_from_fen_with_black_to_move() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
        let mut g = GameState::new(Color::Black, Some(fen)).unwrap();
        play_all(&mut g, &["c5", "Nf3"]);
        assert!(g.pgn().contains("1... c5 2. Nf3"));
        assert!(GameState::new(Color::White, Some("garbage")).is_err());
    }

    #[test]
    fn repetition() {
        let mut g = GameState::new(Color::White, None).unwrap();
        play_all(&mut g, &["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]);
        assert_eq!(g.outcome().unwrap().reason, "threefold repetition");
    }
}
