//! The engine actor: one task that owns the Stockfish process and runs the UCI
//! conversation with it.
//!
//! Invariant: every `go` or `stop` we send is followed by reading until its
//! `bestmove`. A process where that read fails (timeout, EOF, bad output) is killed
//! and never used again, so a late answer can never be read as the reply to a later
//! question.
//!
//! The engine's stdout is drained by a dedicated reader task that forwards lines
//! over a channel. The actor consumes that channel even while idle, so a long
//! ponder can never fill the OS pipe and block the engine on its own output.
//!
//! While a search runs, the actor keeps listening for requests. `Stop`, `NewGame`
//! and `Shutdown` cancel the search (`stop`, then read its `bestmove`), so they
//! never wait for the search to finish. Every request is still handled in the order
//! it was sent.

use std::collections::VecDeque;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use cg_core::Score;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{timeout, timeout_at, Instant};

use crate::{uci, EngineConfig, EngineError, EngineReply, Position, Request};

/// How long to wait for a killed process to exit.
const KILL_WAIT: Duration = Duration::from_secs(2);

/// Owns the process, the request queue and the ponder state.
pub(crate) struct Actor {
    cfg: EngineConfig,
    proc: Option<Proc>,
    /// The ponder search currently running, if any.
    ponder: Option<Ponder>,
    name: String,
    rx: mpsc::UnboundedReceiver<Request>,
    /// Requests that arrived while a search was running, in arrival order.
    pending: VecDeque<Request>,
    /// False once every [`crate::Engine`] handle is dropped.
    rx_open: bool,
}

/// A running engine process and its pipes.
struct Proc {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    /// Lines from the reader task. Closed when the engine closes its stdout.
    lines: mpsc::UnboundedReceiver<std::io::Result<String>>,
    reader: JoinHandle<()>,
}

/// A ponder search in progress, with the latest exact `multipv 1` score and depth
/// seen so far. Only these are kept; the other lines are dropped as they arrive.
struct Ponder {
    pos: Position,
    progress: Progress,
}

/// The latest exact `multipv 1` score and depth of a search.
#[derive(Clone, Copy, Default)]
struct Progress {
    score: Option<Score>,
    depth: u32,
}

impl Progress {
    /// Update from an `info` line, ignoring other PV lines and bound scores.
    fn track(&mut self, line: &str) {
        let Some(info) = uci::parse_info(line) else {
            return;
        };
        if info.multipv == 1 && !info.bound {
            if let Some(d) = info.depth {
                self.depth = d;
            }
            if info.score.is_some() {
                self.score = info.score;
            }
        }
    }
}

/// The result of one search, before timing is added.
struct Search {
    bestmove: String,
    ponder: Option<String>,
    progress: Progress,
}

/// What the idle loop woke up for.
enum Event {
    Request(Request),
    Line(Result<String, EngineError>),
    Closed,
}

/// Process the requests until every handle is dropped or a shutdown arrives.
pub(crate) async fn run(mut actor: Actor) {
    loop {
        let req = match actor.next_event().await {
            Event::Request(req) => req,
            Event::Line(line) => {
                actor.on_idle_line(line).await;
                continue;
            }
            Event::Closed => break,
        };
        match req {
            Request::BestMove { pos, reply } => {
                let res = actor.best_move(&pos).await;
                let _ = reply.send(res);
            }
            Request::Ponder { pos } => actor.ponder(pos).await,
            Request::Stop { ack } => {
                actor.stop().await;
                let _ = ack.send(());
            }
            Request::NewGame { ack } => {
                actor.new_game().await;
                let _ = ack.send(());
            }
            Request::Shutdown { ack } => {
                actor.kill().await;
                let _ = ack.send(());
                return;
            }
        }
    }
    actor.kill().await;
}

impl Request {
    /// Requests that end a running search early instead of waiting behind it.
    fn cancels_search(&self) -> bool {
        matches!(
            self,
            Request::Stop { .. } | Request::NewGame { .. } | Request::Shutdown { .. }
        )
    }
}

impl Actor {
    pub(crate) fn new(cfg: EngineConfig, rx: mpsc::UnboundedReceiver<Request>) -> Self {
        Actor {
            cfg,
            proc: None,
            ponder: None,
            name: String::new(),
            rx,
            pending: VecDeque::new(),
            rx_open: true,
        }
    }

    /// Engine name from the most recent handshake.
    pub(crate) fn name(&self) -> String {
        self.name.clone()
    }

    /// Spawn the process and handshake now, so start-up errors surface in `Engine::start`.
    pub(crate) async fn connect(&mut self) -> Result<(), EngineError> {
        self.ensure_proc().await.map(|_| ())
    }

    /// The next request (queued ones first), or an engine line that arrived while idle.
    async fn next_event(&mut self) -> Event {
        if let Some(req) = self.pending.pop_front() {
            return Event::Request(req);
        }
        if !self.rx_open {
            return Event::Closed;
        }
        let Some(proc) = self.proc.as_mut() else {
            return match self.rx.recv().await {
                Some(req) => Event::Request(req),
                None => Event::Closed,
            };
        };
        tokio::select! {
            req = self.rx.recv() => match req {
                Some(req) => Event::Request(req),
                None => Event::Closed,
            },
            line = proc.lines.recv() => Event::Line(line_result(line)),
        }
    }

    /// An engine line read between requests: the progress of a ponder search, or
    /// noise. Anything that breaks the protocol kills the process.
    async fn on_idle_line(&mut self, line: Result<String, EngineError>) {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                tracing::warn!(error = %e, "engine output failed while idle; killing the process");
                self.kill().await;
                return;
            }
        };
        tracing::trace!(target: "cg_engine::io", "<< {line}");
        let bestmove = line.starts_with("bestmove");
        match self.ponder.as_mut() {
            // A ponder search only answers after `ponderhit` or `stop`.
            Some(_) if bestmove => {
                tracing::warn!("engine answered a ponder search unasked; killing the process");
                self.kill().await;
            }
            Some(p) => p.progress.track(&line),
            None if bestmove => tracing::warn!("unexpected bestmove while idle: {line}"),
            None => {}
        }
    }

    /// The live process, spawning and handshaking one if there is none.
    async fn ensure_proc(&mut self) -> Result<&mut Proc, EngineError> {
        if self.proc.is_none() {
            let mut proc = Proc::spawn(&self.cfg.path).await?;
            self.name = handshake(&mut proc, &self.cfg).await?;
            tracing::debug!(name = %self.name, "engine process ready");
            self.proc = Some(proc);
        }
        Ok(self.proc_mut())
    }

    fn proc_mut(&mut self) -> &mut Proc {
        self.proc.as_mut().expect("engine process is running")
    }

    /// Kill the process (if any) and forget any ponder state.
    async fn kill(&mut self) {
        self.ponder = None;
        if let Some(proc) = self.proc.take() {
            proc.kill().await;
        }
    }

    /// True if a request that cancels searches is waiting.
    fn cancel_pending(&self) -> bool {
        self.pending.iter().any(Request::cancels_search)
    }

    /// Best move for `pos`, with one restart-and-retry on timeout or crash.
    async fn best_move(&mut self, pos: &Position) -> Result<EngineReply, EngineError> {
        let started = Instant::now();
        let mut retried = false;
        loop {
            match self.search(pos, started).await {
                Ok(reply) => return Ok(reply),
                // A finished game or a cancelled search is a valid answer; the process
                // is still clean (its `bestmove` was read).
                Err(e @ (EngineError::GameOver | EngineError::Cancelled)) => return Err(e),
                Err(e) => {
                    tracing::warn!(error = %e, "engine search failed; killing the process");
                    self.kill().await;
                    // Nobody wants the answer any more, so don't retry.
                    if self.cancel_pending() {
                        return Err(EngineError::Cancelled);
                    }
                    let retryable = matches!(e, EngineError::Timeout | EngineError::Crashed(_));
                    if retried || !retryable {
                        return Err(e);
                    }
                    retried = true;
                }
            }
        }
    }

    /// One attempt: ponderhit if `pos` was being pondered, otherwise a normal search.
    async fn search(&mut self, pos: &Position, started: Instant) -> Result<EngineReply, EngineError> {
        let io = self.cfg.handshake_timeout;
        let mut progress = Progress::default();
        let ponder_hit = self.ponder.as_ref().is_some_and(|p| p.pos == *pos);
        if ponder_hit {
            if let Some(p) = self.ponder.take() {
                progress = p.progress;
            }
            self.ensure_proc().await?.send("ponderhit", io).await?;
        } else {
            self.stop_ponder().await?;
            let movetime = self.cfg.movetime.as_millis();
            let proc = self.ensure_proc().await?;
            proc.send_position(pos, io).await?;
            proc.send(&format!("go movetime {movetime}"), io).await?;
        }
        // Stockfish counts `movetime` from the original `go`, so a ponder hit normally
        // answers at once. The deadline still bounds a hung process.
        let deadline = Instant::now() + self.cfg.movetime + self.cfg.slack;
        let search = self.read_search(deadline, progress, true).await?;
        let search = search.ok_or(EngineError::GameOver)?;
        Ok(EngineReply {
            bestmove: search.bestmove,
            ponder: search.ponder,
            score: search.progress.score.unwrap_or(Score::Cp(0)),
            depth: search.progress.depth,
            elapsed: started.elapsed(),
            ponder_hit,
        })
    }

    /// Stop a running ponder search and read (discard) its `bestmove`.
    async fn stop_ponder(&mut self) -> Result<(), EngineError> {
        if self.ponder.take().is_none() {
            return Ok(());
        }
        let io = self.cfg.handshake_timeout;
        let deadline = Instant::now() + self.cfg.slack;
        let Some(proc) = self.proc.as_mut() else {
            return Ok(());
        };
        proc.send("stop", io).await?;
        self.read_search(deadline, Progress::default(), false).await?;
        Ok(())
    }

    /// Start pondering `pos`. Replaces any earlier ponder.
    async fn ponder(&mut self, pos: Position) {
        if let Err(e) = self.start_ponder(&pos).await {
            tracing::warn!(error = %e, "ponder failed; killing the process");
            self.kill().await;
        }
    }

    async fn start_ponder(&mut self, pos: &Position) -> Result<(), EngineError> {
        self.stop_ponder().await?;
        let io = self.cfg.handshake_timeout;
        let movetime = self.cfg.movetime.as_millis();
        let proc = self.ensure_proc().await?;
        proc.send_position(pos, io).await?;
        proc.send(&format!("go ponder movetime {movetime}"), io).await?;
        self.ponder = Some(Ponder {
            pos: pos.clone(),
            progress: Progress::default(),
        });
        Ok(())
    }

    /// Stop any ponder. A failure kills the process.
    async fn stop(&mut self) {
        if let Err(e) = self.stop_ponder().await {
            tracing::warn!(error = %e, "stop failed; killing the process");
            self.kill().await;
        }
    }

    /// `ucinewgame` + `isready`. With no process running, the next one starts clean.
    async fn new_game(&mut self) {
        if let Err(e) = self.stop_ponder().await {
            tracing::warn!(error = %e, "stop before new game failed; killing the process");
            self.kill().await;
            return;
        }
        let io = self.cfg.handshake_timeout;
        let Some(proc) = self.proc.as_mut() else {
            return;
        };
        let res = async {
            proc.send("ucinewgame", io).await?;
            proc.send("isready", io).await?;
            proc.wait_for("readyok", Instant::now() + io).await
        }
        .await;
        if let Err(e) = res {
            tracing::warn!(error = %e, "ucinewgame failed; killing the process");
            self.kill().await;
        }
    }

    /// Read until `bestmove`, keeping the last exact `multipv 1` score and depth
    /// (starting from `progress`). Returns `None` for `bestmove (none)`.
    ///
    /// With `cancellable`, requests are queued while reading; one that cancels
    /// searches sends `stop`, and the search then ends with `Cancelled` once its
    /// `bestmove` has been read.
    async fn read_search(
        &mut self,
        mut deadline: Instant,
        mut progress: Progress,
        cancellable: bool,
    ) -> Result<Option<Search>, EngineError> {
        let io = self.cfg.handshake_timeout;
        let slack = self.cfg.slack;
        let mut cancelled = false;
        loop {
            let listen = cancellable && !cancelled && self.rx_open;
            let proc = self.proc.as_mut().expect("engine process is running");
            let line = tokio::select! {
                biased;
                line = proc.read_line(deadline) => line?,
                req = self.rx.recv(), if listen => {
                    match req {
                        Some(req) => {
                            let cancels = req.cancels_search();
                            self.pending.push_back(req);
                            if cancels {
                                tracing::debug!("cancelling the running search");
                                cancelled = true;
                                deadline = deadline.min(Instant::now() + slack);
                                self.proc_mut().send("stop", io).await?;
                            }
                        }
                        // Every handle is gone; let the search finish for its caller.
                        None => self.rx_open = false,
                    }
                    continue;
                }
            };
            if let Some(rest) = line.strip_prefix("bestmove") {
                if cancelled {
                    return Err(EngineError::Cancelled);
                }
                if rest.trim_start().starts_with("(none)") {
                    return Ok(None);
                }
                let (bestmove, ponder) = uci::parse_bestmove(rest)?;
                return Ok(Some(Search {
                    bestmove,
                    ponder,
                    progress,
                }));
            }
            progress.track(&line);
        }
    }
}

/// Map a reader-channel item to a line or the error it stands for.
fn line_result(item: Option<std::io::Result<String>>) -> Result<String, EngineError> {
    match item {
        Some(Ok(line)) => Ok(line),
        Some(Err(e)) => Err(EngineError::Crashed(format!("read failed: {e}"))),
        None => Err(EngineError::Crashed("engine closed its output".into())),
    }
}

impl Proc {
    async fn spawn(path: &Path) -> Result<Proc, EngineError> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| match e.kind() {
                ErrorKind::NotFound => EngineError::NotInstalled,
                _ => EngineError::Io(e),
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EngineError::Crashed("engine has no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EngineError::Crashed("engine has no stdout".into()))?;
        // Drain stdout continuously so the engine never blocks writing to it.
        let (tx, lines) = mpsc::unbounded_channel();
        let reader = tokio::spawn(async move {
            let mut stdout = BufReader::new(stdout).lines();
            loop {
                let item = stdout.next_line().await.transpose();
                let last = !matches!(item, Some(Ok(_)));
                if let Some(item) = item {
                    if tx.send(item).is_err() {
                        return;
                    }
                }
                if last {
                    return;
                }
            }
        });
        Ok(Proc {
            child,
            stdin: BufWriter::new(stdin),
            lines,
            reader,
        })
    }

    /// Write one command line. Fails with `Timeout` if the pipe blocks for `io`.
    async fn send(&mut self, line: &str, io: Duration) -> Result<(), EngineError> {
        tracing::debug!(target: "cg_engine::io", ">> {line}");
        let write = async {
            self.stdin.write_all(line.as_bytes()).await?;
            self.stdin.write_all(b"\n").await?;
            self.stdin.flush().await
        };
        match timeout(io, write).await {
            Err(_) => Err(EngineError::Timeout),
            Ok(Err(e)) => Err(EngineError::Crashed(format!("write failed: {e}"))),
            Ok(Ok(())) => Ok(()),
        }
    }

    async fn send_position(&mut self, pos: &Position, io: Duration) -> Result<(), EngineError> {
        let mut line = match &pos.start_fen {
            None => "position startpos".to_string(),
            Some(fen) => format!("position fen {fen}"),
        };
        if !pos.moves.is_empty() {
            line.push_str(" moves ");
            line.push_str(&pos.moves.join(" "));
        }
        self.send(&line, io).await
    }

    /// The next line from the reader task. Cancel-safe.
    async fn read_line(&mut self, deadline: Instant) -> Result<String, EngineError> {
        match timeout_at(deadline, self.lines.recv()).await {
            Err(_) => Err(EngineError::Timeout),
            Ok(item) => {
                let line = line_result(item)?;
                tracing::debug!(target: "cg_engine::io", "<< {line}");
                Ok(line)
            }
        }
    }

    /// Read lines until one equals `token`.
    async fn wait_for(&mut self, token: &str, deadline: Instant) -> Result<(), EngineError> {
        loop {
            if self.read_line(deadline).await?.trim() == token {
                return Ok(());
            }
        }
    }

    /// Kill the process and wait for it to exit.
    async fn kill(mut self) {
        let _ = self.child.start_kill();
        let _ = timeout(KILL_WAIT, self.child.wait()).await;
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// `uci` / `uciok`, options, `isready` / `readyok`. Returns the engine's id name.
async fn handshake(proc: &mut Proc, cfg: &EngineConfig) -> Result<String, EngineError> {
    let io = cfg.handshake_timeout;
    proc.send("uci", io).await?;
    let deadline = Instant::now() + io;
    let mut name = String::new();
    loop {
        let line = proc.read_line(deadline).await?;
        if let Some(n) = line.strip_prefix("id name ") {
            name = n.trim().to_string();
        } else if line.trim() == "uciok" {
            break;
        }
    }
    proc.send(&format!("setoption name Threads value {}", cfg.threads), io)
        .await?;
    proc.send(&format!("setoption name Hash value {}", cfg.hash_mb), io)
        .await?;
    proc.send("isready", io).await?;
    proc.wait_for("readyok", Instant::now() + io).await?;
    Ok(name)
}
