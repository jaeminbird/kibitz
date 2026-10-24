//! A minimal UCI engine for tests. Behaviour is chosen by `FAKE_UCI_MODE`:
//!
//! - `normal`: answers searches after ~20 ms; honours `go ponder`, `ponderhit`, `stop`.
//! - `hang`: never answers `go`.
//! - `crash`: exits on `go`.
//! - `late`: answers `go` after one second with `a7a6`.
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
use std::thread::sleep;
use std::time::Duration;

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

    let mut out = io::stdout().lock();
    // Moves in the position, and in the pending ponder search.
    let mut position_moves = 0usize;
    let mut pondering: Option<usize> = None;

    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();
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
                    pondering = Some(position_moves);
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
                    _ => answer(&mut out, position_moves)?,
                }
            }
            Some("ponderhit") => {
                if let Some(n) = pondering.take() {
                    answer(&mut out, n)?;
                }
            }
            Some("stop") => {
                if let Some(n) = pondering.take() {
                    answer(&mut out, n)?;
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
    if moves.is_multiple_of(2) {
        writeln!(out, "bestmove e2e4 ponder e7e5")
    } else {
        writeln!(out, "bestmove d2d4 ponder d7d5")
    }
}
