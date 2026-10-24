//! Scene stability gate. STUB — see docs/specs/cg-vision.md (Part B).
use std::time::Instant;

pub struct GateConfig {
    pub thumb_w: u32,
    pub thumb_h: u32,
    pub motion_threshold: f32,
    pub settle_ms: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { thumb_w: 64, thumb_h: 36, motion_threshold: 3.0, settle_ms: 300 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateState {
    Moving,
    Settling,
    Stable { fresh: bool },
}

pub struct StabilityGate {
    _cfg: GateConfig,
}

impl StabilityGate {
    pub fn new(cfg: GateConfig) -> Self {
        StabilityGate { _cfg: cfg }
    }
    pub fn push(&mut self, _gray: &[u8], _width: u32, _height: u32, _at: Instant) -> GateState {
        todo!()
    }
    pub fn motion(&self) -> f32 {
        todo!()
    }
    pub fn reset(&mut self) {
        todo!()
    }
}
