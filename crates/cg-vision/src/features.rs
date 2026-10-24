//! Per-square appearance features and their difference. STUB — see docs/specs/cg-vision.md (Part A).
use crate::geometry::WarpedBoard;

pub struct SquareStats {
    pub cells: [[f32; 16]; 64],
    pub std: [f32; 64],
}

pub fn square_stats(_b: &WarpedBoard) -> SquareStats {
    todo!()
}

pub fn square_diff(_reference: &SquareStats, _current: &SquareStats) -> [f32; 64] {
    todo!()
}
