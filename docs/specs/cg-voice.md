# cg-voice spec

Speech output with near-zero start latency. Deps: `cg-core`, `rodio` (0.23), `tracing`, `anyhow`/`thiserror`, `sha2` (cache keys). Synthesis uses the platform TTS through `std::process::Command`.

## Why
The original spawned macOS `say` per phrase, adding hundreds of ms. Here the closed vocabulary from `cg_core::speech::vocabulary()` (pieces, "to", "takes", 64 square names, …) is rendered once to WAV files. Speaking a move then means concatenating cached PCM clips in memory and handing them to an already-open audio output stream.

## API
```rust
pub trait Speaker: Send + Sync {
    /// Non-blocking. A new phrase interrupts whatever is playing.
    fn say(&self, text: &str);
}

pub struct NullSpeaker;          // does nothing
pub struct LogSpeaker;           // tracing::info!("say: {text}")

pub struct VoiceConfig {
    pub cache_dir: PathBuf,           // e.g. data_dir/voice
    pub device: Option<String>,       // case-insensitive substring of the output device name ("BlackHole", "CABLE Input"); None = default device
    pub voice: Option<String>,        // platform voice name
    pub gap_ms: u32,                  // silence between words, default 40
}

pub struct ClipSpeaker;
impl ClipSpeaker {
    /// Opens the output device and starts the playback thread. Fails if `device` is not found
    /// (error message lists available devices).
    pub fn new(cfg: VoiceConfig) -> anyhow::Result<ClipSpeaker>;
    /// Ensure a clip exists for every word (synthesize missing ones in ONE batch call)
    /// and load all clips into memory. Blocking; call from a background thread at startup.
    pub fn warm(&self, words: &[String]) -> anyhow::Result<()>;
}
impl Speaker for ClipSpeaker { … }

/// Synthesize several (text, wav_path) pairs with the platform TTS, efficiently:
/// - macOS: `say -o <path> --data-format=LEI16@22050 [-v voice] <text>`; may run in parallel threads.
/// - Windows: ONE `powershell -NoProfile -NonInteractive -Command` script using System.Speech
///   (SpeechSynthesizer, SetOutputToWaveFile(path, SpeechAudioFormatInfo(22050, Sixteen, Mono)))
///   that loops over all items. Pass the items via a temp JSON file, not the command line.
/// - Linux: `espeak-ng -w <path> <text>`.
pub fn synthesize_batch(items: &[(String, PathBuf)], voice: Option<&str>) -> anyhow::Result<()>;

pub fn list_output_devices() -> Vec<String>;
```

## Behaviour details
- Clip file name: words matching `^[A-Za-z0-9]+$` → `<word>.wav`. Anything else → `phrase-<first 16 hex chars of sha256>.wav`.
- On load: decode the WAV (rodio decoder or a tiny PCM16 reader), convert to mono f32 at one common sample rate, and **trim** leading/trailing silence (|s| < 0.01), keeping 10 ms.
- `say(text)`: split on whitespace. If every token has a loaded clip, concatenate with `gap_ms` silence and play. Otherwise synthesize the whole phrase as one clip (cached for next time) on the playback thread, then play it. Never block the caller.
- Interrupt: a new `say` stops the current sound (e.g. clear the rodio `Sink`/`Player`) before appending.
- rodio 0.23's API differs from older versions you may remember (output stream/device builders, mixer, `Sink`/`Player`). Read the source under `~/.cargo/registry/src/*/rodio-0.23*/` (and its cpal dependency for device enumeration and names) and get it compiling on Windows. It must stay cross-platform.
- The audio output stream object may be `!Send`. Keep it on the playback thread, which owns it for life. The `ClipSpeaker` handle only holds a channel sender.

## Tests
- Pure tests: file-name mapping, silence trimming, concatenation length = Σ clips + gaps.
- Synthesis test (Windows, which this machine is): `synthesize_batch` for ["Knight", "e4"] into a temp dir. Both WAVs exist, are non-empty, and decode. Mark `#[ignore]` if it takes more than a few seconds, but run it once yourself and report the result.
- `list_output_devices()` returns without panicking. Print the list in your report.
- A manual `examples/say.rs`: `cargo run -p cg-voice --example say -- "Knight to f3"` warms the vocabulary and speaks. Run it once (the sound plays on the default device; that's fine) and report the warm time and the time from `say()` to the audio being appended.
- Use `CARGO_TARGET_DIR=target/agent-voice`.

## Rules
Edit only `crates/cg-voice/**`. If the contract must change, report it instead.
