# cg-engine spec

Drives Stockfish over UCI as a **persistent** process with pondering. Deps you may add to `crates/cg-engine/Cargo.toml` (use `workspace = true` for any in the root `[workspace.dependencies]`): `tokio`, `thiserror`, `tracing`, `shakmaty`, `sha2`, `dirs`, `reqwest` (download only), `zip`/`flate2`/`tar` if needed for unpacking (add these to the crate, pinned).

## Public API (`src/lib.rs` re-exports)
```rust
pub struct EngineConfig {
    pub path: std::path::PathBuf,
    pub movetime: std::time::Duration,      // default 300 ms
    pub threads: u32,                        // default: available_parallelism - 1, min 1
    pub hash_mb: u32,                        // default 128
    pub handshake_timeout: std::time::Duration, // default 10 s
    pub slack: std::time::Duration,          // default 2 s: extra wait beyond movetime before "timeout"
}
impl EngineConfig { pub fn new(path: PathBuf) -> Self }  // defaults above

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Position { pub start_fen: Option<String>, pub moves: Vec<String> /* UCI */ }

#[derive(Clone, Debug)]
pub struct EngineReply {
    pub bestmove: String,            // UCI
    pub ponder: Option<String>,      // UCI expected reply
    pub score: cg_core::Score,       // side-to-move POV (last "info … score" for multipv 1)
    pub depth: u32,
    pub elapsed: std::time::Duration,
    pub ponder_hit: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("stockfish is not installed; run `kibitz setup`")] NotInstalled,
    #[error("the game is over")] GameOver,
    #[error("engine timed out")] Timeout,
    #[error("search cancelled")] Cancelled,
    #[error("engine crashed: {0}")] Crashed(String),
    #[error("protocol error: {0}")] Protocol(String),
    #[error("invalid position: {0}")] BadPosition(String),
    #[error(transparent)] Io(#[from] std::io::Error),
}

#[derive(Clone)]
pub struct Engine { /* mpsc sender to the actor task */ }
impl Engine {
    /// Spawn the process, handshake (uci→uciok, setoption Threads/Hash/Ponder=true, isready→readyok).
    pub async fn start(cfg: EngineConfig) -> Result<Engine, EngineError>;
    /// Best move for `pos`. "Steady" semantics: on Timeout or Crashed, kill the process,
    /// start a fresh one, retry ONCE; a second failure returns the error.
    /// Checks the position with shakmaty first: illegal → BadPosition, no legal moves → GameOver.
    /// A stop/new_game/shutdown sent while the search runs ends it early → Cancelled.
    pub fn best_move(&self, pos: Position) -> impl Future<Output = Result<EngineReply, EngineError>> + Send + 'static;
    /// Start pondering `pos` (normally: position after our suggested move + its expected reply),
    /// using `go ponder movetime <movetime>`. Returns immediately. Replaces any previous ponder.
    pub fn ponder(&self, pos: Position);
    /// Stop any ponder (discarding its result) and cancel a running best_move. Used on undo/new game.
    pub fn stop(&self) -> impl Future<Output = ()> + Send + 'static;
    /// "ucinewgame" + isready; clears hash. Cancels a running search like `stop`.
    pub fn new_game(&self) -> impl Future<Output = ()> + Send + 'static;
    // best_move/ponder/stop/new_game queue their request when called, not when the future is
    // first polled, so calls made in order reach the engine in order even if awaited elsewhere.
    /// Engine id name from the handshake, e.g. "Stockfish 17.1".
    pub fn name(&self) -> String;
    pub async fn shutdown(self);
}

pub fn data_dir() -> PathBuf;                    // dirs::data_dir()/kibitz
/// Override env KIBITZ_STOCKFISH (used as-is) → `stockfish` on PATH (stockfish.exe on Windows)
/// → data_dir()/bin/stockfish[.exe]. Return NotInstalled if none exists.
pub fn locate() -> Result<PathBuf, EngineError>;
/// Download a pinned official build for this OS/arch, verify SHA-256, unpack the binary to
/// data_dir()/bin/stockfish[.exe] via a ".part" file + rename, mark it executable (unix).
pub async fn download(progress: impl FnMut(u64, Option<u64>) + Send) -> anyhow::Result<PathBuf>;
```

## Actor design (critical)
- One tokio task owns the `Child`, a `BufWriter<ChildStdin>` and the receiving end of a line channel. Requests come over an mpsc channel; each `best_move` carries a oneshot for the reply.
- **Stdout is always drained**: a dedicated reader task per process reads stdout lines and forwards them over the line channel. The actor consumes that channel even between requests, so a long ponder (Stockfish writes `info` lines continuously, and `currmove` lines after 3 s) can never fill the OS pipe and block the engine. While pondering, only the latest exact `multipv 1` score and depth are kept; a ponder hit starts from them. A `bestmove` while pondering, or EOF while idle, kills the process.
- **Cancellation**: while a search is being read, the actor keeps receiving requests and queues them in order. A `Stop`, `NewGame` or `Shutdown` sends `stop` to the engine, reads the `bestmove` (deadline at most `slack`), answers the search with `Cancelled`, and then handles the queued requests. A cancelled search is not retried.
- **Ponder state**: `Idle | Pondering { pos, score, depth }`.
  - `best_move(pos)` while `Pondering{p}` and `p == pos`: send `ponderhit`, then read until `bestmove`, reply with `ponder_hit: true`. (Stockfish counts movetime from the original `go`, so a long ponder answers immediately.)
  - `best_move(pos)` while pondering something else: send `stop`, read and discard until its `bestmove`, then do a normal search.
  - Normal search: `position [startpos|fen F] moves …` then `go movetime N`, and read `info` lines (keep the last `score cp X` / `score mate Y` and `depth`, ignoring lines with `multipv` > 1 and `lowerbound`/`upperbound` lines) until `bestmove X [ponder Y]`.
- **Timeouts**: each read loop has a deadline of `movetime + slack` (handshake: `handshake_timeout`). On timeout or EOF, kill the child (`start_kill` + wait) and mark it dead. **Never reuse a timed-out process**: a late `bestmove` must never be read as the answer to a later question. The retry spawns a brand-new process and handshakes again.
- A `bestmove (none)` reply → GameOver.
- The actor must not block on stdin writes forever; use the async writer.
- Log with `tracing` at debug level: every line sent and received.

## Download pins
Find the latest official Stockfish release on GitHub (`curl -s https://api.github.com/repos/official-stockfish/Stockfish/releases/latest`). Pin these assets as constants:
- Windows x86-64: the `stockfish-windows-x86-64-avx2.zip` asset
- macOS: the apple-silicon asset (aarch64) and the x86-64 asset
- Linux x86-64 avx2
Download each pinned asset once with curl, compute SHA-256 (`sha256sum`), and hard-code the hash. If the network is unavailable, leave the hashes as `None` and make `download` refuse to install without a hash ("no pinned checksum").

## Tests
- Add a fake engine binary `src/bin/fake-uci.rs` (in this crate) that speaks minimal UCI. Behaviour is controlled by env `FAKE_UCI_MODE`: `normal` (answers `bestmove e2e4 ponder e7e5` with `info depth 10 score cp 31` after ~20 ms, honours `go ponder`/`ponderhit`/`stop`), `hang` (never answers `go`), `crash` (exits on `go`), `hang-once` (hangs the first time it is launched, using a marker file in a temp dir from env `FAKE_UCI_MARKER`, then behaves normally), `flood` (a ponder search writes info lines nonstop until `ponderhit`/`stop`, then answers with a bare `bestmove`, or `bestmove a2a3` if a write blocked ≥ 200 ms), `slow` (a normal `go` runs 30 s unless stopped).
- Integration tests in `tests/engine.rs` use `env!("CARGO_BIN_EXE_fake-uci")`:
  - normal best move; score parsed; depth parsed
  - ponder then matching best_move → `ponder_hit == true` and fast
  - ponder then different best_move → correct answer, `ponder_hit == false`
  - `hang-once` → best_move succeeds via restart (retry once)
  - `hang` → Timeout after one retry, within ~2×(movetime+slack)
  - `crash` → Crashed
  - game-over position (checkmate FEN) → GameOver without touching the engine
  - `flood` → a long, noisy ponder does not stall; the ponder hit keeps the pondered score and depth
  - `slow` → stop / new_game / shutdown during a long search return promptly; the search returns Cancelled
  - stop / new_game / ponder called before best_move are handled first, even when best_move is awaited first
- An `#[ignore]` test that runs a real Stockfish from `locate()` if one is present.
- Use `CARGO_TARGET_DIR=target/agent-engine` for all cargo commands.

## Rules
Edit only `crates/cg-engine/**`. Do not edit cg-core. If the contract seems wrong, report it instead of changing it.
