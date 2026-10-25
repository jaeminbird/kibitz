//! Golden data for the Python ports in `tools/kibitz_tools/board.py`.
//!
//! Not part of the normal test run. Regenerate with
//!
//! ```text
//! cargo test -p cg-vision --test golden -- --ignored
//! ```
//!
//! which writes `tools/tests/golden/`: two synthetic frames (start position,
//! and after 1. e4 with an exposure change), the corners, the warped boards and
//! the square stats / diff at two square sizes. `tools/tests/test_parity.py`
//! checks the Python ports against these files.

use std::fmt::Write as _;
use std::path::Path;

use cg_core::Corners;
use cg_vision::{square_diff, square_stats, synth, warp_board, SquareStats};
use shakmaty::{uci::UciMove, Chess, Position};

const W: u32 = 320;
const H: u32 = 240;
/// 32 is the runtime size (`cg-app/src/vision.rs`). 30 makes the 15% margin
/// land on 4.5, which checks that the port rounds half away from zero like Rust.
const SQ_PX: [u32; 2] = [32, 30];

/// Binary PGM (P5), 8-bit.
fn write_pgm(path: &Path, w: u32, h: u32, data: &[u8]) {
    let mut out = format!("P5\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(data);
    std::fs::write(path, out).unwrap();
}

/// `{"cells": [[16 floats] × 64], "std": [64 floats]}`.
fn stats_json(s: &SquareStats) -> String {
    let rows: Vec<String> = s.cells.iter().map(|c| format!("[{}]", join(c))).collect();
    format!("{{\"cells\": [{}], \"std\": [{}]}}", rows.join(", "), join(&s.std))
}

fn join(v: &[f32]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")
}

#[test]
#[ignore = "writes golden files for tools/; run explicitly"]
fn write_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/tests/golden");
    std::fs::create_dir_all(&dir).unwrap();

    // Perspective quad with the h1 corner pushed past the right edge, so the
    // out-of-frame = 0 rule is exercised too.
    let corners = Corners([[70.0, 30.5], [262.3, 33.0], [331.0, 228.0], [20.0, 221.7]]);
    let mut pos = Chess::default();
    let before = pos.board().clone();
    let mv = "e2e4".parse::<UciMove>().unwrap().to_move(&pos).unwrap();
    pos.play_unchecked(mv);

    let p_a = synth::SynthParams { corners, seed: 11, ..synth::default_params(W, H) };
    let p_b = synth::SynthParams { corners, seed: 12, brightness: 8.0, ..synth::default_params(W, H) };
    let frame_a = synth::render(&before, &p_a);
    let frame_b = synth::render(pos.board(), &p_b);
    write_pgm(&dir.join("frame_a.pgm"), W, H, &frame_a);
    write_pgm(&dir.join("frame_b.pgm"), W, H, &frame_b);

    let mut json = String::new();
    let c = corners.0.map(|[x, y]| format!("[{x}, {y}]")).join(", ");
    write!(json, "{{\"width\": {W}, \"height\": {H}, \"corners\": [{c}], \"cases\": [").unwrap();
    for (i, sq) in SQ_PX.into_iter().enumerate() {
        let wa = warp_board(&frame_a, W, H, &corners, sq).unwrap();
        let wb = warp_board(&frame_b, W, H, &corners, sq).unwrap();
        let n = wa.size();
        write_pgm(&dir.join(format!("warped_a_{sq}.pgm")), n, n, &wa.img.data);
        write_pgm(&dir.join(format!("warped_b_{sq}.pgm")), n, n, &wb.img.data);
        let (sa, sb) = (square_stats(&wa), square_stats(&wb));
        let diff = square_diff(&sa, &sb);
        if i > 0 {
            json.push_str(", ");
        }
        write!(
            json,
            "{{\"sq_px\": {sq}, \"stats_a\": {}, \"stats_b\": {}, \"diff\": [{}]}}",
            stats_json(&sa),
            stats_json(&sb),
            join(&diff)
        )
        .unwrap();
    }
    json.push_str("]}\n");
    std::fs::write(dir.join("golden.json"), json).unwrap();
}
