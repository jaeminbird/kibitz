//! A minimal UCI engine for tests. Behaviour is chosen by `FAKE_UCI_MODE`:
//!
//! - `normal`: answers searches after ~20 ms; honours `go ponder`, `ponderhit`, `stop`.
//! - `hang`: never answers `go`.
//! - `crash`: exits on `go`.
//! - `late`: answers `go` after one second with `a7a6`.
//! - `flood`: like `normal`, but a ponder search writes `info` lines nonstop (exact
//!   `multipv 1` lines with `score cp 42`, plus `currmove` lines) until `ponderhit`
//!   or `stop`, and then answers with a bare `bestmove`. If nobody reads stdout the
//!   pipe fills and the writes block, which stalls a real engine's search; when a
//!   write blocked for 200 ms or more, the answer is `bestmove a2a3` instead.
//! - `slow`: a normal `go` searches for 30 s, writing an `info` line every 5 ms, and
//!   only answers early on `stop`. Ponder searches behave as in `normal`.
//! - `hang-once` / `late-once`: the first launch behaves like `hang` / `late`; later
//!   launches are `normal`. "First launch" means `FAKE_UCI_MARKER` does not exist yet;
//!   the launch creates it.
//!
//! Answers depend on the number of moves in the position: an even count gives
//! `e2e4 ponder e7e5`, an odd count gives `d2d4 ponder d7d5`.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

/// A flood write that blocks this long means nobody was reading our output, so the
/// "search" stalled.
const STALL: Duration = Duration::from_millis(200);

/// A background search thread that runs until told to stop. It returns true if
/// it stalled on a blocked write.
struct Worker {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<io::Result<bool>>,
}

impl Worker {
    fn spawn(body: impl FnOnce(&AtomicBool) -> io::Result<bool> + Send + 'static) -> Worker {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || body(&flag));
        Worker { stop, handle }
    }

    /// Signal the thread and wait for it. Blocks while the thread is stuck writing.
    fn finish(self) -> io::Result<bool> {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.join().expect("worker thread panicked")
    }
}

fn main() -> io::Result<()> {
    let requested = std::env::var("FAKE_UCI_MODE").unwrap_or_else(|_| "normal".into());
    let mode = match requested.strip_suffix("-once") {
        Some(base) => {
            let first = match std::env::var_os("FAKE_UCI_MARKER").map(PathBuf::from) {
                Some(marker) if !marker.exists() => {
                    fs::write(&marker, b"launched")?;
                    true
                }
                _ => false,
            };
            if first { base.to_string() } else { "normal".to_string() }
        }
        None => requested,
    };

    // Moves in the position, and in the pending ponder search (with its flood
    // thread, if any).
    let mut position_moves = 0usize;
    let mut pondering: Option<(usize, Option<Worker>)> = None;
    // A `slow` search in progress; it prints its own `bestmove`.
    let mut searching: Option<Worker> = None;

    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();
        let mut out = io::stdout();
        match tokens.next() {
            Some("uci") => {
                writeln!(out, "id name Fake UCI 1.0")?;
                writeln!(out, "id author kibitz tests")?;
                writeln!(out, "uciok")?;
            }
            Some("isready") => writeln!(out, "readyok")?,
            Some("position") => {
                position_moves = line
                    .split_whitespace()
                    .skip_while(|t| *t != "moves")
                    .skip(1)
                    .count();
            }
            Some("go") => {
                if line.split_whitespace().any(|t| t == "ponder") {
                    let flood = (mode == "flood").then(|| Worker::spawn(flood));
                    pondering = Some((position_moves, flood));
                    continue;
                }
                match mode.as_str() {
                    "hang" => loop {
                        sleep(Duration::from_secs(3600));
                    },
                    "crash" => exit(1),
                    "late" => {
                        sleep(Duration::from_secs(1));
                        writeln!(out, "info depth 1 score cp 0")?;
                        writeln!(out, "bestmove a7a6")?;
                    }
                    "slow" => {
                        let n = position_moves;
                        searching = Some(Worker::spawn(move |stop| slow_search(stop, n)));
                    }
                    _ => answer(&mut out, position_moves)?,
                }
            }
            Some("ponderhit") | Some("stop") => {
                if let Some((n, flood)) = pondering.take() {
                    match flood {
                        // A stalled search answers with a move it would never
                        // otherwise pick, so tests can tell.
                        Some(worker) => {
                            if worker.finish()? {
                                writeln!(out, "bestmove a2a3")?;
                            } else {
                                bestmove(&mut out, n)?;
                            }
                        }
                        None => answer(&mut out, n)?,
                    }
                }
                if let Some(worker) = searching.take() {
                    worker.finish()?;
                }
            }
            Some("quit") => break,
            _ => {}
        }
        out.flush()?;
    }
    Ok(())
}

/// Print a search result for a position with `moves` moves played from the start.
fn answer(out: &mut impl Write, moves: usize) -> io::Result<()> {
    sleep(Duration::from_millis(20));
    writeln!(out, "info depth 9 score cp 99 upperbound")?;
    writeln!(out, "info depth 10 multipv 2 score cp -200")?;
    writeln!(out, "info depth 10 seldepth 12 multipv 1 score cp 31 nodes 900 pv e2e4")?;
    bestmove(out, moves)
}

fn bestmove(out: &mut impl Write, moves: usize) -> io::Result<()> {
    if moves.is_multiple_of(2) {
        writeln!(out, "bestmove e2e4 ponder e7e5")
    } else {
        writeln!(out, "bestmove d2d4 ponder d7d5")
    }
}

/// Write ponder output as fast as a busy engine would, until stopped. Returns true
/// if a write blocked for `STALL` or longer.
fn flood(stop: &AtomicBool) -> io::Result<bool> {
    let mut depth = 1u32;
    let mut nodes = 0u64;
    let mut stalled = false;
    while !stop.load(Ordering::SeqCst) {
        let started = Instant::now();
        let mut out = io::stdout().lock();
        for i in 0..8 {
            nodes += 1000;
            writeln!(
                out,
                "info depth {depth} seldepth {} multipv 1 score cp 42 nodes {nodes} nps 1000000 \
                 pv e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
                depth + 4
            )?;
            writeln!(out, "info depth {depth} currmove e2e4 currmovenumber {}", i + 1)?;
        }
        out.flush()?;
        drop(out);
        stalled |= started.elapsed() >= STALL;
        depth += 1;
        sleep(Duration::from_millis(1));
    }
    Ok(stalled)
}

/// A long normal search that only ends early on `stop`.
fn slow_search(stop: &AtomicBool, moves: usize) -> io::Result<bool> {
    let started = Instant::now();
    let mut depth = 1u32;
    while !stop.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(30) {
        writeln!(io::stdout(), "info depth {depth} multipv 1 score cp 7")?;
        depth += 1;
        sleep(Duration::from_millis(5));
    }
    let mut out = io::stdout().lock();
    bestmove(&mut out, moves)?;
    out.flush()?;
    Ok(false)
}
