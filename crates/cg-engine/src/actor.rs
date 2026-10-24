//! The engine actor: one task that owns the Stockfish process and runs the UCI
//! conversation with it.
//!
//! Invariant: every `go` or `stop` we send is followed by reading until its
//! `bestmove`. A process where that read fails (timeout, EOF, bad output) is killed
//! and never used again, so a late answer can never be read as the reply to a later
//! question.

use std::io::ErrorKind;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use cg_core::Score;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::time::{timeout, timeout_at, Instant};

use crate::{uci, EngineConfig, EngineError, EngineReply, Position, Request};

/// How long to wait for a killed process to exit.
const KILL_WAIT: Duration = Duration::from_secs(2);

/// Owns the process and the ponder state.
pub(crate) struct Actor {
    cfg: EngineConfig,
    proc: Option<Proc>,
    /// The position the engine is currently pondering, if any.
    ponder: Option<Position>,
    name: String,
}

/// A running engine process and its pipes.
struct Proc {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: Lines<BufReader<ChildStdout>>,
}

/// The result of one search, before timing is added.
struct Search {
    bestmove: String,
    ponder: Option<String>,
    score: Option<Score>,
    depth: u32,
}

/// Process the requests until every handle is dropped or a shutdown arrives.
pub(crate) async fn run(mut actor: Actor, mut rx: mpsc::UnboundedReceiver<Request>) {
    while let Some(req) = rx.recv().await {
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

impl Actor {
    pub(crate) fn new(cfg: EngineConfig) -> Self {
        Actor {
            cfg,
            proc: None,
            ponder: None,
            name: String::new(),
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

    /// Best move for `pos`, with one restart-and-retry on timeout or crash.
    async fn best_move(&mut self, pos: &Position) -> Result<EngineReply, EngineError> {
        let started = Instant::now();
        let mut retried = false;
        loop {
            match self.search(pos, started).await {
                Ok(reply) => return Ok(reply),
                // A finished game is a valid answer; the process is still clean.
                Err(EngineError::GameOver) => return Err(EngineError::GameOver),
                Err(e) => {
                    tracing::warn!(error = %e, "engine search failed; killing the process");
                    self.kill().await;
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
        let ponder_hit = self.ponder.as_ref() == Some(pos);
        if ponder_hit {
            self.ponder = None;
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
        let search = self.proc_mut().read_search(deadline).await?;
        let search = search.ok_or(EngineError::GameOver)?;
        Ok(EngineReply {
            bestmove: search.bestmove,
            ponder: search.ponder,
            score: search.score.unwrap_or(Score::Cp(0)),
            depth: search.depth,
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
        proc.read_search(deadline).await?;
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
        self.ponder = Some(pos.clone());
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
        Ok(Proc {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout).lines(),
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

    async fn read_line(&mut self, deadline: Instant) -> Result<String, EngineError> {
        match timeout_at(deadline, self.stdout.next_line()).await {
            Err(_) => Err(EngineError::Timeout),
            Ok(Ok(Some(line))) => {
                tracing::debug!(target: "cg_engine::io", "<< {line}");
                Ok(line)
            }
            Ok(Ok(None)) => Err(EngineError::Crashed("engine closed its output".into())),
            Ok(Err(e)) => Err(EngineError::Crashed(format!("read failed: {e}"))),
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

    /// Read until `bestmove`, keeping the last `multipv 1` score and depth. Returns
    /// `None` for `bestmove (none)`.
    async fn read_search(&mut self, deadline: Instant) -> Result<Option<Search>, EngineError> {
        let mut score = None;
        let mut depth = 0;
        loop {
            let line = self.read_line(deadline).await?;
            if let Some(info) = uci::parse_info(&line) {
                if info.multipv == 1 && !info.bound {
                    if let Some(d) = info.depth {
                        depth = d;
                    }
                    if info.score.is_some() {
                        score = info.score;
                    }
                }
            } else if let Some(rest) = line.strip_prefix("bestmove") {
                if rest.trim_start().starts_with("(none)") {
                    return Ok(None);
                }
                let (bestmove, ponder) = uci::parse_bestmove(rest)?;
                return Ok(Some(Search {
                    bestmove,
                    ponder,
                    score,
                    depth,
                }));
            }
        }
    }

    /// Kill the process and wait for it to exit.
    async fn kill(mut self) {
        let _ = self.child.start_kill();
        let _ = timeout(KILL_WAIT, self.child.wait()).await;
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
