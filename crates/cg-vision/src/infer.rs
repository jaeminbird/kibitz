//! Legal-move inference from square diffs. STUB — see docs/specs/cg-vision.md (Part A).
use shakmaty::{Chess, Move};

pub struct InferConfig {
    pub min_change: f32,
    pub min_margin: f32,
    pub max_residual: f32,
}

impl Default for InferConfig {
    fn default() -> Self {
        todo!()
    }
}

pub struct Inference {
    pub best: Option<Move>,
    pub best_uci: Option<String>,
    pub score: f32,
    pub margin: f32,
    pub residual: f32,
    pub changed: u32,
    pub ranked: Vec<(String, f32)>,
    pub accept: bool,
    pub reason: Option<&'static str>,
}

pub fn changed_squares(_pos: &Chess, _mv: &Move) -> Vec<u8> {
    todo!()
}

pub fn infer_move(_pos: &Chess, _diff: &[f32; 64], _cfg: &InferConfig) -> Inference {
    todo!()
}
