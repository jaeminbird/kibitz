//! Warms the vocabulary and speaks one phrase on the default output device.
//!
//! `cargo run -p cg-voice --example say -- "Knight to f3"`

use std::time::{Duration, Instant};

use cg_core::speech::vocabulary;
use cg_voice::{ClipSpeaker, Speaker, VoiceConfig};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("cg_voice=debug")
        .with_target(false)
        .init();

    let phrase = std::env::args().nth(1).unwrap_or_else(|| "Knight to f3".into());
    let cfg = VoiceConfig {
        cache_dir: std::env::temp_dir().join("cg-voice-example"),
        device: None,
        voice: None,
        gap_ms: 40,
    };

    let t = Instant::now();
    let speaker = ClipSpeaker::new(cfg)?;
    println!("open device: {:?}", t.elapsed());

    let words = vocabulary();
    let t = Instant::now();
    speaker.warm(&words)?;
    println!("warm {} words: {:?}", words.len(), t.elapsed());

    let t = Instant::now();
    speaker.say(&phrase);
    println!("say() returned after {:?}", t.elapsed());

    // Clips are well under 3 s; give playback time to finish before exiting.
    std::thread::sleep(Duration::from_secs(3));
    Ok(())
}
