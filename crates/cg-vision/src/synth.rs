//! Synthetic board renderer for tests and benchmarks. STUB — see docs/specs/cg-vision.md (Part A).
use cg_core::Corners;

pub struct SynthParams {
    pub width: u32,
    pub height: u32,
    pub corners: Corners,
    pub noise: f32,
    pub brightness: f32,
    pub seed: u64,
}

pub fn render(_board: &shakmaty::Board, _p: &SynthParams) -> Vec<u8> {
    todo!()
}

pub fn default_params(_width: u32, _height: u32) -> SynthParams {
    todo!()
}

pub fn render_hand(_img: &mut [u8], _width: u32, _height: u32, _cx: f32, _cy: f32, _r: f32) {
    todo!()
}
