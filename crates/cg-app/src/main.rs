//! `chessglass`: reads an over-the-board chess game from a video call,
//! picks the best move with Stockfish, and speaks it into the call.
//! For friendly games where everyone at the board knows; not for rated play.

mod bench;
mod doctor;
mod game;
mod orchestrator;
mod setup;
mod vision;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use tokio::sync::{mpsc, watch};

#[derive(Parser)]
#[command(name = "chessglass", version, about = "Low-latency chess move reader and adviser")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorArg {
    White,
    Black,
}

#[derive(Subcommand)]
enum Cmd {
    /// Watch a game and speak suggestions.
    Run(RunArgs),
    /// Download Stockfish, pre-render voice clips, store API keys.
    Setup {
        #[arg(long)]
        key_anthropic: Option<String>,
        #[arg(long)]
        key_openai: Option<String>,
        /// Output device for the voice warm-up check.
        #[arg(long)]
        voice_device: Option<String>,
    },
    /// Check every dependency and say how to fix what's missing.
    Doctor {
        #[arg(long, default_value = "window:WhatsApp")]
        source: String,
    },
    /// Measure pipeline latency on synthetic frames (and Stockfish if installed).
    Bench {
        #[arg(long, default_value_t = 200)]
        frames: u32,
        /// Instead of benchmarking, write a synthetic game as PNG frames (for `run --source dir:`).
        #[arg(long)]
        dump: Option<PathBuf>,
        /// With --dump: script the game from Stockfish self-play.
        #[arg(long)]
        selfplay: bool,
    },
}

#[derive(clap::Args)]
struct RunArgs {
    /// Your color.
    #[arg(long, value_enum, default_value = "white")]
    color: ColorArg,
    /// Start from this position instead of the initial one.
    #[arg(long)]
    fen: Option<String>,
    /// window:<title> | file:<video> | dir:<frames>
    #[arg(long, default_value = "window:WhatsApp")]
    source: String,
    #[arg(long, default_value_t = 15.0)]
    fps: f32,
    /// Engine think time when not answering from a ponder hit.
    #[arg(long, default_value_t = 300)]
    movetime_ms: u64,
    /// How long the board must be still before it is read.
    #[arg(long, default_value_t = 300)]
    settle_ms: u32,
    #[arg(long)]
    no_ponder: bool,
    /// Vision LLM used only to find the board / break ties.
    #[arg(long, env = "CHESSGLASS_MODEL", default_value = cg_llm::DEFAULT_MODEL)]
    model: String,
    /// Never call a paid vision service (calibrate the board by clicking its corners).
    #[arg(long)]
    no_llm: bool,
    /// Pause when vision spending reaches this many USD.
    #[arg(long, default_value_t = 10.0)]
    max_cost: f64,
    /// Output device substring for speech (e.g. "BlackHole", "CABLE Input").
    #[arg(long)]
    voice_device: Option<String>,
    #[arg(long)]
    no_voice: bool,
    /// Save game.pgn and key frames here.
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// Append every event as JSON lines to this file.
    #[arg(long)]
    log: Option<PathBuf>,
    /// Don't open the command center in a browser.
    #[arg(long)]
    no_open: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();
    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        match cli.cmd {
            Cmd::Run(args) => run(args).await,
            Cmd::Setup { key_anthropic, key_openai, voice_device } => {
                setup::run(key_anthropic, key_openai, voice_device).await
            }
            Cmd::Doctor { source } => doctor::run(&source).await,
            Cmd::Bench { frames, dump: Some(dir), selfplay } => bench::dump(&dir, frames, selfplay).await,
            Cmd::Bench { frames, dump: None, .. } => bench::run(frames).await,
        }
    })
}

async fn run(args: RunArgs) -> anyhow::Result<()> {
    let my_color = match args.color {
        ColorArg::White => shakmaty::Color::White,
        ColorArg::Black => shakmaty::Color::Black,
    };
    if let Some(dir) = &args.run_dir {
        std::fs::create_dir_all(dir.join("frames")).context("creating run folder")?;
    }
    let bus = cg_core::Bus::new(args.log.as_deref()).context("opening event log")?;

    // Engine first: it is the one hard dependency.
    let sf = cg_engine::locate().context("Stockfish not found; run `chessglass setup`")?;
    let mut ecfg = cg_engine::EngineConfig::new(sf);
    ecfg.movetime = Duration::from_millis(args.movetime_ms);
    let engine = cg_engine::Engine::start(ecfg).await.context("starting Stockfish")?;
    tracing::info!("engine: {}", engine.name());

    let llm = if args.no_llm {
        None
    } else {
        let provider = cg_llm::Provider::for_model(&args.model);
        match cg_llm::resolve_key(provider) {
            Some(key) => Some(Arc::new(cg_llm::VisionLlm::new(cg_llm::LlmConfig::new(&args.model, key)))),
            None => {
                tracing::warn!("no API key for {provider:?}; board must be calibrated by clicking its corners");
                None
            }
        }
    };

    let speaker: Arc<dyn cg_voice::Speaker> = if args.no_voice {
        Arc::new(cg_voice::LogSpeaker)
    } else {
        let cfg = cg_voice::VoiceConfig {
            cache_dir: cg_engine::data_dir().join("voice"),
            device: args.voice_device.clone(),
            voice: None,
            gap_ms: 40,
        };
        let clip = Arc::new(cg_voice::ClipSpeaker::new(cfg).context("opening audio output")?);
        let warm = clip.clone();
        std::thread::spawn(move || {
            if let Err(e) = warm.warm(&cg_core::speech::vocabulary()) {
                tracing::warn!("voice warm-up failed: {e:#}");
            }
        });
        clip
    };

    let (frame_tx, frame_rx) = watch::channel(None);
    let (status_tx, status_rx) = mpsc::unbounded_channel();
    let (control_tx, control_rx) = mpsc::channel(64);
    let stop = Arc::new(AtomicBool::new(false));
    let source = cg_capture::open_source(&args.source, args.fps).context("opening capture source")?;
    tracing::info!("capturing from {}", source.describe());
    let capture = cg_capture::spawn_capture(source, frame_tx, status_tx, stop.clone());

    let server = cg_server::serve(cg_server::ServerConfig::default(), bus.clone(), frame_rx.clone(), control_tx)
        .await
        .context("starting command center")?;
    let url = format!("http://{}/", server.addr);
    tracing::info!("command center: {url}");
    if !args.no_open {
        open_browser(&url);
    }

    let cfg = orchestrator::Config {
        my_color,
        fen: args.fen,
        settle_ms: args.settle_ms,
        ponder: !args.no_ponder,
        max_cost: args.max_cost,
        run_dir: args.run_dir,
    };
    let deps = orchestrator::Deps { bus, engine, llm, speaker };
    let result = orchestrator::run(cfg, deps, frame_rx, control_rx, status_rx).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = capture.join();
    result
}

fn open_browser(url: &str) {
    let cmd = if cfg!(target_os = "windows") {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Err(e) = cmd {
        tracing::warn!("couldn't open a browser ({e}); visit {url}");
    }
}
