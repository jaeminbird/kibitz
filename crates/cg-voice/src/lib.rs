//! Speech output with near-zero start latency. The closed vocabulary is rendered
//! once to WAV clips; speaking a phrase concatenates cached clips in memory and
//! hands them to an already-open audio output on a dedicated playback thread.

mod clip;
mod synth;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use rodio::cpal::traits::HostTrait;
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, DeviceTrait, MixerDeviceSink, Player};

pub use clip::{clip_file_name, concat_clips, load_clip, trim_silence, CLIP_RATE};
pub use synth::synthesize_batch;

/// Speaks phrases. Implementations must never block the caller.
pub trait Speaker: Send + Sync {
    /// Non-blocking. A new phrase interrupts whatever is playing.
    fn say(&self, text: &str);
}

/// Does nothing.
pub struct NullSpeaker;

impl Speaker for NullSpeaker {
    fn say(&self, _text: &str) {}
}

/// Logs phrases instead of speaking them.
pub struct LogSpeaker;

impl Speaker for LogSpeaker {
    fn say(&self, text: &str) {
        tracing::info!("say: {text}");
    }
}

/// Settings for [`ClipSpeaker`].
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Directory holding the rendered WAV clips, e.g. `data_dir/voice`.
    pub cache_dir: PathBuf,
    /// Case-insensitive substring of the output device name; `None` = default device.
    pub device: Option<String>,
    /// Platform voice name; `None` = platform default.
    pub voice: Option<String>,
    /// Silence between words, in milliseconds.
    pub gap_ms: u32,
}

/// Plays pre-rendered word clips on a named output device.
pub struct ClipSpeaker {
    /// Phrase text plus the instant `say` was called, for latency logging.
    tx: Sender<(Instant, String)>,
    shared: Arc<Shared>,
}

/// State shared between the handle (warming) and the playback thread (playing).
struct Shared {
    /// Loaded word clips, keyed by the exact word.
    clips: RwLock<HashMap<String, Arc<[f32]>>>,
    cfg: VoiceConfig,
}

impl ClipSpeaker {
    /// Opens the output device and starts the playback thread. Fails if `device` is not found
    /// (the error lists the available devices).
    pub fn new(cfg: VoiceConfig) -> Result<ClipSpeaker> {
        let shared = Arc::new(Shared {
            clips: RwLock::new(HashMap::new()),
            cfg: cfg.clone(),
        });
        let (tx, rx) = mpsc::channel::<(Instant, String)>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<()>>(1);
        let thread_shared = shared.clone();
        std::thread::Builder::new()
            .name("cg-voice-playback".into())
            .spawn(move || {
                // The output stream is opened and kept on this thread for its whole life.
                let sink = match open_sink(cfg.device.as_deref()) {
                    Ok(sink) => sink,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(()));
                playback_loop(sink, rx, thread_shared);
            })
            .context("spawning playback thread")?;
        ready_rx
            .recv()
            .context("playback thread exited during startup")??;
        Ok(ClipSpeaker { tx, shared })
    }

    /// Ensure a clip exists for every word (synthesizing missing ones in ONE batch call)
    /// and load all clips into memory. Blocking; call from a background thread at startup.
    pub fn warm(&self, words: &[String]) -> Result<()> {
        let cfg = &self.shared.cfg;
        std::fs::create_dir_all(&cfg.cache_dir)
            .with_context(|| format!("creating {}", cfg.cache_dir.display()))?;

        let unique: HashSet<&str> = words.iter().map(String::as_str).collect();
        let mut missing = Vec::new();
        for word in &unique {
            let path = cfg.cache_dir.join(clip_file_name(word));
            if !path.exists() {
                missing.push((word.to_string(), path));
            }
        }
        if !missing.is_empty() {
            tracing::info!("synthesizing {} missing clips", missing.len());
            synthesize_batch(&missing, cfg.voice.as_deref())?;
        }

        let mut loaded = HashMap::with_capacity(unique.len());
        for word in unique {
            let path = cfg.cache_dir.join(clip_file_name(word));
            let samples = load_clip(&path)?;
            if samples.is_empty() {
                tracing::warn!("clip for {word:?} is silent");
            }
            loaded.insert(word.to_string(), Arc::from(samples));
        }
        self.shared
            .clips
            .write()
            .expect("clip lock poisoned")
            .extend(loaded);
        Ok(())
    }
}

impl Speaker for ClipSpeaker {
    fn say(&self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        if self.tx.send((Instant::now(), text.to_string())).is_err() {
            tracing::error!("playback thread is gone; dropping {text:?}");
        }
    }
}

/// Owns the output stream. Each phrase gets a fresh `Player`; dropping the previous one
/// stops its sound, which is how a new phrase interrupts the current one.
fn playback_loop(sink: MixerDeviceSink, rx: Receiver<(Instant, String)>, shared: Arc<Shared>) {
    // Holds the playing phrase; replacing it drops (and stops) the previous one.
    let mut _playing: Option<Player> = None;
    while let Ok((mut said_at, mut text)) = rx.recv() {
        // Only the newest phrase matters; older ones would be interrupted anyway.
        while let Ok((at, newer)) = rx.try_recv() {
            said_at = at;
            text = newer;
        }
        _playing = None;
        match shared.utterance(&text) {
            Ok(samples) if samples.is_empty() => {}
            Ok(samples) => {
                let player = Player::connect_new(sink.mixer());
                player.append(SamplesBuffer::new(
                    std::num::NonZero::new(1).expect("1 is nonzero"),
                    std::num::NonZero::new(CLIP_RATE).expect("rate is nonzero"),
                    samples,
                ));
                player.play();
                tracing::debug!(
                    "queued {text:?} {:?} after say()",
                    said_at.elapsed()
                );
                _playing = Some(player);
            }
            Err(e) => tracing::error!("cannot speak {text:?}: {e:#}"),
        }
    }
}

impl Shared {
    /// Builds the samples for a phrase: concatenated word clips when every word is
    /// loaded, otherwise the whole phrase rendered as one (cached) clip.
    fn utterance(&self, text: &str) -> Result<Vec<f32>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let gap = (CLIP_RATE as usize) * (self.cfg.gap_ms as usize) / 1000;
        let found: Option<Vec<Arc<[f32]>>> = {
            let clips = self.clips.read().expect("clip lock poisoned");
            words.iter().map(|w| clips.get(*w).cloned()).collect()
        };
        match found {
            Some(list) => {
                let refs: Vec<&[f32]> = list.iter().map(|c| &c[..]).collect();
                Ok(concat_clips(&refs, gap))
            }
            None => self.phrase_clip(text),
        }
    }

    /// Renders (or reads from cache) a whole phrase as one clip.
    fn phrase_clip(&self, text: &str) -> Result<Vec<f32>> {
        let path = self.cfg.cache_dir.join(clip_file_name(text));
        if !path.exists() {
            synthesize_batch(&[(text.to_string(), path.clone())], self.cfg.voice.as_deref())?;
        }
        load_clip(&path)
    }
}

/// Opens the named output device (case-insensitive substring), or the default one.
fn open_sink(device: Option<&str>) -> Result<MixerDeviceSink> {
    let mut sink = match device {
        None => DeviceSinkBuilder::open_default_sink()?,
        Some(needle) => {
            let needle_lc = needle.to_lowercase();
            let dev = rodio::cpal::default_host()
                .output_devices()
                .context("listing output devices")?
                .find(|d| device_name(d).to_lowercase().contains(&needle_lc))
                .ok_or_else(|| {
                    anyhow!(
                        "no output device matches {needle:?}; available: {}",
                        list_output_devices().join(", ")
                    )
                })?;
            DeviceSinkBuilder::from_device(dev)?.open_stream()?
        }
    };
    sink.log_on_drop(false);
    Ok(sink)
}

fn device_name(dev: &rodio::Device) -> String {
    dev.description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unnamed>".into())
}

/// Names of the available audio output devices. Empty if the host cannot enumerate them.
pub fn list_output_devices() -> Vec<String> {
    match rodio::cpal::default_host().output_devices() {
        Ok(devices) => devices.map(|d| device_name(&d)).collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_and_log_speakers_accept_anything() {
        NullSpeaker.say("Knight to f3");
        LogSpeaker.say("");
    }

    #[test]
    fn list_output_devices_does_not_panic() {
        let devices = list_output_devices();
        eprintln!("output devices: {devices:#?}");
    }

    #[test]
    fn unknown_device_is_an_error_listing_alternatives() {
        let cfg = VoiceConfig {
            cache_dir: std::env::temp_dir().join("cg-voice-unused"),
            device: Some("no-such-device-xyz-123".into()),
            voice: None,
            gap_ms: 40,
        };
        let err = match ClipSpeaker::new(cfg) {
            Ok(_) => panic!("unknown device should fail"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("no-such-device-xyz-123"), "{err}");
        assert!(err.contains("available"), "{err}");
    }
}
