//! Board corner tracker and checkerboard validation. STUB — see docs/specs/cg-vision.md (Part B).
use cg_core::Corners;

use crate::geometry::WarpedBoard;

pub struct TrackerConfig {
    pub coarse_scale: u32,
    pub search_px: i32,
    pub patch_px: u32,
    pub refine_px: i32,
    pub min_checker: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig { coarse_scale: 8, search_px: 64, patch_px: 48, refine_px: 10, min_checker: 0.35 }
    }
}

pub struct CornerTracker {
    _cfg: TrackerConfig,
}

impl CornerTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        CornerTracker { _cfg: cfg }
    }
    pub fn reset(&mut self, _gray: &[u8], _width: u32, _height: u32, _corners: Corners) {
        todo!()
    }
    pub fn track(&mut self, _gray: &[u8], _width: u32, _height: u32) -> Option<Corners> {
        todo!()
    }
    pub fn corners(&self) -> Option<Corners> {
        todo!()
    }
    pub fn last_score(&self) -> f32 {
        todo!()
    }
}

pub fn checker_score(_b: &WarpedBoard) -> f32 {
    todo!()
}
