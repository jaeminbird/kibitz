//! Integration tests against the `fake-uci` binary, plus an ignored test that
//! runs a real Stockfish (`cargo test -p cg-engine -- --ignored`).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use cg_core::Score;
use cg_engine::{Engine, EngineConfig, EngineError, Position};
use tokio::sync::{Mutex, MutexGuard};

/// The fake engine reads its mode from the environment when it is spawned, and a
/// restart spawns it again, so tests that set the environment must not overlap.
static ENV_LOCK: Mutex<()> = Mutex::const_new(());

/// Holds the environment lock and a temp dir for the marker file.
struct TestEnv {
    _guard: MutexGuard<'static, ()>,
    _dir: tempfile::TempDir,
}

async fn setup(mode: &str) -> TestEnv {
    let guard = ENV_LOCK.lock().await;
    let dir = tempfile::tempdir().expect("temp dir");
    std::env::set_var("FAKE_UCI_MODE", mode);
    std::env::set_var("FAKE_UCI_MARKER", dir.path().join("launched"));
    TestEnv {
        _guard: guard,
        _dir: dir,
    }
}

/// Short timings: a search attempt gives up after 50 ms + 300 ms.
fn config() -> EngineConfig {
    let mut cfg = EngineConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_fake-uci")));
    cfg.movetime = Duration::from_millis(50);
    cfg.slack = Duration::from_millis(300);
    cfg.handshake_timeout = Duration::from_secs(5);
    cfg.threads = 1;
    cfg.hash_mb = 16;
    cfg
}

fn from_start(moves: &[&str]) -> Position {
    Position {
        start_fen: None,
        moves: moves.iter().map(|m| m.to_string()).collect(),
    }
}

#[tokio::test]
async fn normal_best_move_has_score_depth_and_ponder() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");
    assert!(engine.name().contains("Fake UCI"), "name: {}", engine.name());

    let r = engine.best_move(from_start(&[])).await.expect("best move");
    assert_eq!(r.bestmove, "e2e4");
    assert_eq!(r.ponder.as_deref(), Some("e7e5"));
    assert_eq!(r.score, Score::Cp(31));
    assert_eq!(r.depth, 10);
    assert!(!r.ponder_hit);
    engine.shutdown().await;
}

#[tokio::test]
async fn ponder_then_matching_best_move_is_a_ponder_hit() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");

    // After e2e4 the fake predicts d2d4.
    let pos = from_start(&["e2e4"]);
    engine.ponder(pos.clone()).await;
    let r = engine.best_move(pos.clone()).await.expect("best move");
    assert!(r.ponder_hit);
    assert_eq!(r.bestmove, "d2d4");
    assert_eq!(r.score, Score::Cp(31));

    // A ponder result is used once. The same question again is a normal search.
    let again = engine.best_move(pos).await.expect("second best move");
    assert!(!again.ponder_hit);
    engine.shutdown().await;
}

#[tokio::test]
async fn ponder_then_different_best_move_searches_again() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");

    engine.ponder(from_start(&["e2e4"])).await;
    // The opponent played something else, so the answer comes from a normal search.
    let r = engine
        .best_move(from_start(&["e2e4", "e7e5"]))
        .await
        .expect("best move");
    assert!(!r.ponder_hit);
    assert_eq!(r.bestmove, "e2e4");
    engine.shutdown().await;
}

#[tokio::test]
async fn stop_discards_the_ponder() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");

    engine.ponder(from_start(&["e2e4"])).await;
    engine.stop().await;
    engine.new_game().await;
    let r = engine
        .best_move(from_start(&["e2e4"]))
        .await
        .expect("best move");
    assert!(!r.ponder_hit);
    assert_eq!(r.bestmove, "d2d4");
    engine.shutdown().await;
}

#[tokio::test]
async fn hang_once_restarts_and_answers() {
    let _env = setup("hang-once").await;
    // The first process hangs on `go`, so this search times out and the retry
    // starts a fresh process that answers.
    let engine = Engine::start(config()).await.expect("start");
    let started = Instant::now();
    let r = engine.best_move(from_start(&[])).await.expect("retry succeeds");
    assert_eq!(r.bestmove, "e2e4");
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    engine.shutdown().await;
}

#[tokio::test]
async fn hang_times_out_after_one_retry() {
    let _env = setup("hang").await;
    let engine = Engine::start(config()).await.expect("start");
    let cfg = config();
    let attempt = cfg.movetime + cfg.slack;

    let started = Instant::now();
    let err = engine.best_move(from_start(&[])).await.unwrap_err();
    let elapsed = started.elapsed();
    assert!(matches!(err, EngineError::Timeout), "{err:?}");
    // Two attempts, each waiting movetime + slack.
    assert!(elapsed >= attempt * 2, "too fast: {elapsed:?}");
    assert!(elapsed < attempt * 2 + Duration::from_secs(2), "too slow: {elapsed:?}");
    engine.shutdown().await;
}

#[tokio::test]
async fn crash_is_reported_after_one_retry() {
    let _env = setup("crash").await;
    let engine = Engine::start(config()).await.expect("start");
    let err = engine.best_move(from_start(&[])).await.unwrap_err();
    assert!(matches!(err, EngineError::Crashed(_)), "{err:?}");
    engine.shutdown().await;
}

#[tokio::test]
async fn late_answer_from_a_killed_process_is_never_used() {
    let _env = setup("late-once").await;
    // The first process answers `a7a6` after one second. The 350 ms attempt times out,
    // the process is killed, and the fresh process answers e2e4.
    let engine = Engine::start(config()).await.expect("start");
    let r = engine.best_move(from_start(&[])).await.expect("retry succeeds");
    assert_eq!(r.bestmove, "e2e4");

    // Give the killed process time to have answered if it were still alive.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let next = engine
        .best_move(from_start(&["e2e4"]))
        .await
        .expect("next best move");
    assert_eq!(next.bestmove, "d2d4", "stale answer from the killed process");
    assert!(!next.ponder_hit);
    engine.shutdown().await;
}

#[tokio::test]
async fn checkmate_is_game_over_without_touching_the_engine() {
    // With `hang` the engine would time out, so GameOver must come from the pre-check.
    let _env = setup("hang").await;
    let engine = Engine::start(config()).await.expect("start");
    // Fool's mate, White to move.
    let pos = Position {
        start_fen: Some("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3".into()),
        moves: vec![],
    };
    let started = Instant::now();
    let err = engine.best_move(pos).await.unwrap_err();
    assert!(matches!(err, EngineError::GameOver), "{err:?}");
    assert!(started.elapsed() < Duration::from_millis(200));
    engine.shutdown().await;
}

#[tokio::test]
async fn illegal_move_is_a_bad_position() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");
    let err = engine.best_move(from_start(&["e2e5"])).await.unwrap_err();
    assert!(matches!(err, EngineError::BadPosition(_)), "{err:?}");
    engine.shutdown().await;
}

#[tokio::test]
#[ignore = "needs a real Stockfish found by locate()"]
async fn real_stockfish_answers_and_ponder_hits() {
    let path = cg_engine::locate().expect("stockfish not installed");
    let mut cfg = EngineConfig::new(path);
    cfg.movetime = Duration::from_millis(200);
    cfg.hash_mb = 16;
    let engine = Engine::start(cfg).await.expect("start");
    assert!(!engine.name().is_empty());

    let first = engine.best_move(from_start(&[])).await.expect("best move");
    assert!(first.bestmove.len() >= 4, "{first:?}");
    assert!(first.depth > 0);

    // Ponder on the position after our move and the expected reply, then ask for it.
    let mut moves = vec![first.bestmove.clone()];
    moves.extend(first.ponder.clone());
    let pos = Position {
        start_fen: None,
        moves,
    };
    engine.ponder(pos.clone()).await;
    let hit = engine.best_move(pos).await.expect("ponder hit");
    assert!(hit.ponder_hit, "{hit:?}");
    engine.shutdown().await;
}
