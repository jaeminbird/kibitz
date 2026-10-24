//! `chessglass setup`: install Stockfish, pre-render the voice vocabulary, store keys.

use std::io::Write;

pub async fn run(
    key_anthropic: Option<String>,
    key_openai: Option<String>,
    voice_device: Option<String>,
) -> anyhow::Result<()> {
    match cg_engine::locate() {
        Ok(p) => println!("stockfish: already available at {}", p.display()),
        Err(_) => {
            println!("stockfish: downloading pinned build…");
            let path = cg_engine::download(|done, total| {
                match total {
                    Some(t) if t > 0 => print!("\r  {:>3}% ({done}/{t} bytes)", done * 100 / t),
                    _ => print!("\r  {done} bytes"),
                }
                let _ = std::io::stdout().flush();
            })
            .await?;
            println!("\nstockfish: installed at {}", path.display());
        }
    }

    for (provider, key) in [(cg_llm::Provider::Anthropic, key_anthropic), (cg_llm::Provider::OpenAI, key_openai)] {
        if let Some(k) = key {
            cg_llm::store_key(provider, &k)?;
            println!("{provider:?} key: saved to the OS keychain");
        }
    }

    println!("voice: pre-rendering vocabulary…");
    let t = std::time::Instant::now();
    let cfg = cg_voice::VoiceConfig {
        cache_dir: cg_engine::data_dir().join("voice"),
        device: voice_device,
        voice: None,
        gap_ms: 40,
    };
    let speaker = cg_voice::ClipSpeaker::new(cfg)?;
    speaker.warm(&cg_core::speech::vocabulary())?;
    println!("voice: ready in {:.1}s", t.elapsed().as_secs_f32());
    println!("\nnext: `chessglass doctor`, then `chessglass run --color white`");
    Ok(())
}
