//! `chessglass doctor`: check every dependency and print the fix for anything missing.

use std::time::Duration;

struct Report {
    failures: u32,
}

impl Report {
    fn ok(&self, what: &str, detail: &str) {
        println!("  ok    {what}: {detail}");
    }
    fn warn(&self, what: &str, detail: &str, fix: &str) {
        println!("  warn  {what}: {detail}\n        fix: {fix}");
    }
    fn fail(&mut self, what: &str, detail: &str, fix: &str) {
        self.failures += 1;
        println!("  FAIL  {what}: {detail}\n        fix: {fix}");
    }
}

pub async fn run(source: &str) -> anyhow::Result<()> {
    println!("chessglass doctor\n");
    let mut r = Report { failures: 0 };

    // Stockfish: found, starts, answers.
    match cg_engine::locate() {
        Ok(path) => {
            let mut cfg = cg_engine::EngineConfig::new(path.clone());
            cfg.movetime = Duration::from_millis(50);
            match cg_engine::Engine::start(cfg).await {
                Ok(e) => {
                    let pos = cg_engine::Position { start_fen: None, moves: vec![] };
                    match e.best_move(pos).await {
                        Ok(m) => r.ok("stockfish", &format!("{} at {} (startpos → {})", e.name(), path.display(), m.bestmove)),
                        Err(err) => r.fail("stockfish", &err.to_string(), "reinstall with `chessglass setup`"),
                    }
                    e.shutdown().await;
                }
                Err(err) => r.fail("stockfish", &format!("{} won't start: {err}", path.display()), "run `chessglass setup`, or unset CHESSGLASS_STOCKFISH"),
            }
        }
        Err(_) => r.fail("stockfish", "not found", "run `chessglass setup` (or install it and put it on PATH)"),
    }

    // Capture source: can we grab one frame?
    match cg_capture::open_source(source, 5.0) {
        Ok(mut src) => match src.next_frame() {
            Ok(Some(f)) => r.ok("capture", &format!("{} → {}x{}", src.describe(), f.width, f.height)),
            Ok(None) => r.fail("capture", "source is empty", "check the file/dir path"),
            Err(e) => {
                let windows: Vec<String> = cg_capture::list_windows()
                    .into_iter()
                    .filter(|(_, t, w, h)| !t.is_empty() && *w > 200 && *h > 200)
                    .map(|(app, t, _, _)| format!("{app} — {t}"))
                    .take(8)
                    .collect();
                r.fail(
                    "capture",
                    &e.to_string(),
                    &format!(
                        "open the call window (not minimized) or pass --source window:<title>. Visible windows: {}",
                        windows.join("; ")
                    ),
                );
            }
        },
        Err(e) => r.fail("capture", &e.to_string(), "check --source"),
    }

    // Audio output and the virtual cable that feeds the call's microphone.
    let devices = cg_voice::list_output_devices();
    let virtual_dev = devices.iter().find(|d| {
        let d = d.to_lowercase();
        d.contains("blackhole") || d.contains("cable input") || d.contains("vb-audio")
    });
    match virtual_dev {
        Some(d) => r.ok("audio", &format!("virtual device \"{d}\" (pass --voice-device \"{d}\")")),
        None if devices.is_empty() => r.fail("audio", "no output devices", "check your sound settings"),
        None => r.warn(
            "audio",
            &format!("no virtual cable; speech will play on: {}", devices.join(", ")),
            "install BlackHole 2ch (macOS) or VB-CABLE (Windows) and select it as the call's microphone",
        ),
    }

    // ffmpeg is only needed for file sources.
    match std::process::Command::new("ffmpeg").arg("-version").output() {
        Ok(o) if o.status.success() => r.ok("ffmpeg", "available (needed only for --source file:)"),
        _ => r.warn("ffmpeg", "not on PATH", "install ffmpeg to replay recorded clips"),
    }

    // Optional vision LLM keys.
    for (p, var) in [(cg_llm::Provider::Anthropic, "ANTHROPIC_API_KEY"), (cg_llm::Provider::OpenAI, "OPENAI_API_KEY")] {
        match cg_llm::resolve_key(p) {
            Some(_) => r.ok("api key", &format!("{p:?} key found")),
            None => r.warn("api key", &format!("no {p:?} key (optional)"), &format!("set {var} or `chessglass setup --key-…`; without one, calibrate by clicking the board corners")),
        }
    }

    println!();
    if r.failures == 0 {
        println!("all required checks passed");
        Ok(())
    } else {
        anyhow::bail!("{} required check(s) failed", r.failures)
    }
}
