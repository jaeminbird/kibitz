//! `kibitz bench`: replay a synthetic game through the real vision
//! pipeline (with simulated camera drift and sensor noise) and time Stockfish
//! cold vs. ponder-hit. Prints a latency report.

use std::time::{Duration, Instant};

use cg_core::{Corners, Frame};
use cg_vision::synth;
use shakmaty::{Chess, Position};

use crate::game::{self, GameState};
use crate::vision::{Outcome, VisionPipeline};

const W: u32 = 1280;
const H: u32 = 720;
const FRAME_DT: Duration = Duration::from_millis(66);

pub async fn run(frames_budget: u32) -> anyhow::Result<()> {
    println!("kibitz bench — synthetic {W}x{H} frames at 15 fps\n");
    vision_bench(frames_budget);
    engine_bench().await;
    Ok(())
}

/// Write `moves` plies of the scripted game as PNGs: each position held for 10
/// frames, with slight head drift. Prints the calibration corners (view order,
/// fractions) for the first frame so a test can POST them as `set_corners`.
pub async fn dump(dir: &std::path::Path, moves: u32, selfplay: bool) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let script = if selfplay { selfplay_moves(moves as usize).await? } else { scripted_moves(moves as usize) };
    let base = synth::default_params(W, H);
    let mut pos = Chess::default();
    let mut n = 0u32;
    let mut shift = 0.0f32;
    for k in 0..=script.len() {
        for _ in 0..10 {
            let mut p = synth::default_params(W, H);
            p.corners = Corners(base.corners.0.map(|[x, y]| [x + shift, y + shift * 0.4]));
            p.seed = n as u64 * 7919 + 1;
            let gray = synth::render(pos.board(), &p);
            image::GrayImage::from_raw(W, H, gray)
                .expect("buffer size")
                .save(dir.join(format!("{n:05}.png")))?;
            n += 1;
        }
        if let Some(uci) = script.get(k) {
            let mv = game::parse_move(&pos, uci).expect("scripted move is legal");
            pos.play_unchecked(mv);
            shift += 6.0;
        }
    }
    let view = base.corners.to_view(true).map(|[x, y]| [x / W as f32, y / H as f32]);
    println!("wrote {n} frames; moves: {}", script.join(" "));
    println!("{}", serde_json::json!({ "action": "set_corners", "view": view }));
    Ok(())
}

/// Stockfish self-play: each move is the engine's best move and the next is its
/// predicted reply, so replaying it exercises the ponder-hit path.
async fn selfplay_moves(n: usize) -> anyhow::Result<Vec<String>> {
    let mut cfg = cg_engine::EngineConfig::new(cg_engine::locate()?);
    cfg.movetime = Duration::from_millis(300);
    let engine = cg_engine::Engine::start(cfg).await?;
    let mut moves = Vec::new();
    while moves.len() < n {
        let r = engine.best_move(cg_engine::Position { start_fen: None, moves: moves.clone() }).await?;
        moves.push(r.bestmove);
        if let Some(p) = r.ponder {
            moves.push(p);
        }
    }
    moves.truncate(n);
    engine.shutdown().await;
    Ok(moves)
}

/// A deterministic game: always play the legal move with the median UCI string.
fn scripted_moves(n: usize) -> Vec<String> {
    let mut pos = Chess::default();
    let mut out = Vec::new();
    for _ in 0..n {
        let mut moves: Vec<_> = pos.legal_moves().into_iter().collect();
        if moves.is_empty() {
            break;
        }
        moves.sort_by_key(game::uci);
        let mv = moves[moves.len() / 2];
        out.push(game::uci(&mv));
        pos.play_unchecked(mv);
    }
    out
}

fn vision_bench(frames_budget: u32) {
    let base = synth::default_params(W, H);
    let script = scripted_moves(40);
    let mut game = GameState::new(shakmaty::Color::White, None).unwrap();
    let mut vision = VisionPipeline::new(300);
    let t0 = Instant::now();
    let mut clock = t0;
    let mut seq = 0u64;
    let mut frame_times = Vec::new();
    let (mut committed, mut wrong, mut missed) = (0, 0, 0);

    let make_frame = |board: &shakmaty::Board, shift: f32, seq: &mut u64, clock: &mut Instant| {
        let mut p = synth::default_params(W, H);
        p.corners = Corners(base.corners.0.map(|[x, y]| [x + shift, y + shift * 0.4]));
        p.seed = *seq * 7919 + 1;
        p.noise = 3.0;
        *seq += 1;
        *clock += FRAME_DT;
        let mut f = Frame::from_gray(*seq, W, H, synth::render(board, &p));
        f.at = *clock;
        (f, p.corners)
    };

    // Calibrate on the first frame with the true corners (as a manual click would).
    let (first, corners) = make_frame(game.position().board(), 0.0, &mut seq, &mut clock);
    vision.set_corners(&first, corners);

    let mut shift = 0.0f32;
    'outer: for uci in &script {
        // Hold still long enough to settle, read, then make the move with a little head drift.
        let board = game.position().board().clone();
        for _ in 0..8 {
            let (f, _) = make_frame(&board, shift, &mut seq, &mut clock);
            let pos = game.position().clone();
            if let Outcome::Read { inference, .. } = vision.process(&f, &pos) {
                if inference.accept {
                    wrong += 1; // nothing moved, so any accepted move is wrong
                }
            }
            frame_times.push(vision.last_frame_ms);
            if frame_times.len() as u32 >= frames_budget {
                break 'outer;
            }
        }
        let mv = game.parse_move(uci).unwrap();
        let mut after = game.position().clone();
        after.play_unchecked(mv);
        shift += 6.0; // the head drifts ~6 px per move
        let mut read = false;
        for _ in 0..8 {
            let (f, _) = make_frame(after.board(), shift, &mut seq, &mut clock);
            let pos = game.position().clone();
            let out = vision.process(&f, &pos);
            frame_times.push(vision.last_frame_ms);
            if std::env::var("CG_BENCH_DEBUG").is_ok() {
                let k = match &out { Outcome::NoBoard=>"noboard", Outcome::Searching=>"searching", Outcome::Lost=>"LOST", Outcome::Moving=>"moving", Outcome::Settling=>"settling", Outcome::Baseline=>"baseline", Outcome::Stable=>"stable", Outcome::Read{..}=>"READ" };
                eprint!("{k} ");
            }
            if let Outcome::Read { inference, stats } = out {
                if std::env::var("CG_BENCH_DEBUG").is_ok() {
                    eprintln!("{uci}: accept={} best={:?} reason={:?} margin={:.2} residual={:.2} changed={} ranked={:?}",
                        inference.accept, inference.best_uci, inference.reason, inference.margin, inference.residual, inference.changed, &inference.ranked[..inference.ranked.len().min(3)]);
                }
                if inference.accept && !read {
                    read = true;
                    if inference.best_uci.as_deref() == Some(uci.as_str()) {
                        committed += 1;
                    } else {
                        wrong += 1;
                    }
                    vision.commit_reference(stats);
                }
            }
        }
        if std::env::var("CG_BENCH_DEBUG").is_ok() { eprintln!(" <- {uci}"); }
        if !read {
            missed += 1;
            vision.rebaseline();
        }
        game.play(mv).unwrap();
    }

    frame_times.sort_by(f32::total_cmp);
    let pct = |p: f32| frame_times[((frame_times.len() - 1) as f32 * p) as usize];
    println!("vision ({} frames)", frame_times.len());
    println!("  per-frame cost   p50 {:.2} ms   p95 {:.2} ms   max {:.2} ms", pct(0.5), pct(0.95), pct(1.0));
    println!("  moves read       {committed} correct, {missed} missed, {wrong} WRONG");
    println!("  board settle     300 ms (configurable --settle-ms) dominates detect latency\n");
}

async fn engine_bench() {
    let Ok(path) = cg_engine::locate() else {
        println!("engine: Stockfish not installed, skipped (run `kibitz setup`)");
        return;
    };
    let mut cfg = cg_engine::EngineConfig::new(path);
    cfg.movetime = Duration::from_millis(300);
    let engine = match cg_engine::Engine::start(cfg).await {
        Ok(e) => e,
        Err(e) => {
            println!("engine: failed to start: {e}");
            return;
        }
    };
    // Engine vs. engine: each "turn" we time a cold search for our move, then
    // ponder on the expected reply while the "opponent thinks", and time the hit.
    let (mut cold, mut hit) = (Vec::new(), Vec::new());
    let mut moves: Vec<String> = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        let Ok(reply) = engine.best_move(cg_engine::Position { start_fen: None, moves: moves.clone() }).await else { break };
        cold.push(t.elapsed().as_secs_f32() * 1000.0);
        let Some(ponder) = reply.ponder.clone() else { break };
        moves.push(reply.bestmove.clone());
        moves.push(ponder);
        engine.ponder(cg_engine::Position { start_fen: None, moves: moves.clone() }).await;
        tokio::time::sleep(Duration::from_millis(800)).await; // opponent thinking
        let t = Instant::now();
        let Ok(r) = engine.best_move(cg_engine::Position { start_fen: None, moves: moves.clone() }).await else { break };
        if r.ponder_hit {
            hit.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        // Play that answer and the opponent's predicted reply, then go again.
        moves.push(r.bestmove.clone());
        match r.ponder {
            Some(p) => moves.push(p),
            None => break,
        }
    }
    let avg = |v: &[f32]| if v.is_empty() { f32::NAN } else { v.iter().sum::<f32>() / v.len() as f32 };
    println!("engine ({})", engine.name());
    println!("  normal search    avg {:.1} ms over {} (movetime 300 ms)", avg(&cold), cold.len());
    println!("  ponder hit       avg {:.1} ms over {}", avg(&hit), hit.len());
    engine.shutdown().await;
}
