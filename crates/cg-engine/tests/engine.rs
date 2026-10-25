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
    engine.ponder(pos.clone());
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

    engine.ponder(from_start(&["e2e4"]));
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

    engine.ponder(from_start(&["e2e4"]));
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

/// Generous bound for awaits that should finish promptly; only hit on a regression.
const PROMPT: Duration = Duration::from_secs(3);

async fn within<F: std::future::Future>(what: &str, fut: F) -> F::Output {
    tokio::time::timeout(PROMPT, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within {PROMPT:?}"))
}

#[tokio::test]
async fn ponder_with_lots_of_output_does_not_stall() {
    let _env = setup("flood").await;
    let engine = Engine::start(config()).await.expect("start");

    // The fake writes ~1.5 MB/s of info lines while pondering. Long before this
    // sleep ends, an undrained pipe would be full and the fake blocked on its output,
    // which it reports by answering a2a3.
    let pos = from_start(&["e2e4"]);
    engine.ponder(pos.clone());
    tokio::time::sleep(Duration::from_millis(600)).await;
    let r = within("ponder hit", engine.best_move(pos))
        .await
        .expect("best move");
    assert!(r.ponder_hit);
    assert_eq!(r.bestmove, "d2d4", "the ponder search stalled on a full pipe");
    // The fake answers a ponder hit with a bare `bestmove`, so these come from the
    // info lines read while pondering.
    assert_eq!(r.score, Score::Cp(42));
    assert!(r.depth > 1, "depth {}", r.depth);

    // The same again, but the opponent plays something else: the ponder is stopped.
    engine.ponder(from_start(&["e2e4", "e7e5", "g1f3"]));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let miss = within(
        "search after a ponder miss",
        engine.best_move(from_start(&["e2e4", "e7e5"])),
    )
    .await
    .expect("best move");
    assert!(!miss.ponder_hit);
    assert_eq!(miss.bestmove, "e2e4");
    assert_eq!(miss.score, Score::Cp(31));
    within("shutdown", engine.shutdown()).await;
}

/// A config whose searches only end early when stopped.
fn slow_config() -> EngineConfig {
    let mut cfg = config();
    cfg.movetime = Duration::from_secs(20);
    cfg.slack = Duration::from_secs(2);
    cfg
}

#[tokio::test]
async fn stop_cancels_a_long_search_promptly() {
    let _env = setup("slow").await;
    let engine = Engine::start(slow_config()).await.expect("start");

    let search = tokio::spawn(engine.best_move(from_start(&[])));
    tokio::time::sleep(Duration::from_millis(100)).await;
    within("stop during a search", engine.stop()).await;
    let res = within("cancelled search", search).await.expect("join");
    assert!(matches!(res, Err(EngineError::Cancelled)), "{res:?}");

    // `new_game` cancels too.
    let search = tokio::spawn(engine.best_move(from_start(&["e2e4"])));
    tokio::time::sleep(Duration::from_millis(100)).await;
    within("new game during a search", engine.new_game()).await;
    let res = within("cancelled search", search).await.expect("join");
    assert!(matches!(res, Err(EngineError::Cancelled)), "{res:?}");

    // The process is still usable: ponder searches answer at once in this mode.
    let pos = from_start(&["e2e4"]);
    engine.ponder(pos.clone());
    let r = within("ponder hit", engine.best_move(pos))
        .await
        .expect("best move");
    assert!(r.ponder_hit);
    assert_eq!(r.bestmove, "d2d4");

    // Shutdown also cancels a running search.
    let search = tokio::spawn(engine.best_move(from_start(&[])));
    tokio::time::sleep(Duration::from_millis(100)).await;
    within("shutdown during a search", engine.shutdown()).await;
    let res = within("cancelled search", search).await.expect("join");
    assert!(matches!(res, Err(EngineError::Cancelled)), "{res:?}");
}

#[tokio::test]
async fn requests_reach_the_engine_in_call_order() {
    let _env = setup("normal").await;
    let engine = Engine::start(config()).await.expect("start");
    let pos = from_start(&["e2e4"]);

    // The orchestrator fires `stop` / `new_game` and then asks for a move, awaiting
    // each in its own task. The stop must win even though the search is awaited
    // first: had it run after the search, the search would be a ponder hit.
    engine.ponder(pos.clone());
    let stop = engine.stop();
    let search = tokio::spawn(engine.best_move(pos.clone()));
    let r = within("search", search).await.expect("join").expect("best move");
    assert!(!r.ponder_hit, "stop was handled after best_move");
    within("stop", stop).await;

    engine.ponder(pos.clone());
    let new_game = engine.new_game();
    let r = within("search", engine.best_move(pos.clone()))
        .await
        .expect("best move");
    assert!(!r.ponder_hit, "new_game was handled after best_move");
    within("new game", new_game).await;

    // And a ponder sent before a search is seen first.
    engine.ponder(pos.clone());
    let r = within("search", tokio::spawn(engine.best_move(pos)))
        .await
        .expect("join")
        .expect("best move");
    assert!(r.ponder_hit);
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
    engine.ponder(pos.clone());
    let hit = engine.best_move(pos).await.expect("ponder hit");
    assert!(hit.ponder_hit, "{hit:?}");
    engine.shutdown().await;
}
