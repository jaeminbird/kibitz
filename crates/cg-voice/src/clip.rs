//! Clip files and the pure sample-level helpers used to build utterances.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use rodio::Source;
use sha2::{Digest, Sha256};

/// Every clip is stored and played at this rate; the mixer resamples to the device.
pub const CLIP_RATE: u32 = 22_050;

/// Amplitude below which a sample counts as silence when trimming.
const SILENCE: f32 = 0.01;

/// Silence kept around each trimmed clip, in milliseconds.
const TRIM_PAD_MS: usize = 10;

/// File name for a word: alphanumeric words map to `<word>.wav`, anything
/// else to `phrase-<first 16 hex chars of sha256>.wav`.
pub fn clip_file_name(word: &str) -> String {
    if !word.is_empty() && word.chars().all(|c| c.is_ascii_alphanumeric()) {
        format!("{word}.wav")
    } else {
        let digest = Sha256::digest(word.as_bytes());
        let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
        format!("phrase-{hex}.wav")
    }
}

/// Decodes a clip file to mono f32 at [`CLIP_RATE`] with silence trimmed.
pub fn load_clip(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let decoder = rodio::Decoder::new(BufReader::new(file))
        .with_context(|| format!("decoding {}", path.display()))?;
    let channels = decoder.channels().get() as usize;
    let rate = decoder.sample_rate().get();
    let interleaved: Vec<f32> = decoder.collect();
    let mono = downmix(&interleaved, channels);
    let resampled = resample_linear(&mono, rate, CLIP_RATE);
    Ok(trim_silence(&resampled, CLIP_RATE))
}

/// Averages interleaved channels into one.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear-interpolation resampling. Good enough for speech clips.
pub fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = (samples.len() as f64 / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

/// Cuts leading and trailing silence, keeping [`TRIM_PAD_MS`] on each side.
/// Returns an empty vector if the clip is entirely silent.
pub fn trim_silence(samples: &[f32], rate: u32) -> Vec<f32> {
    let first = samples.iter().position(|s| s.abs() >= SILENCE);
    let last = samples.iter().rposition(|s| s.abs() >= SILENCE);
    let (Some(first), Some(last)) = (first, last) else {
        return Vec::new();
    };
    let pad = rate as usize * TRIM_PAD_MS / 1000;
    let start = first.saturating_sub(pad);
    let end = (last + 1 + pad).min(samples.len());
    samples[start..end].to_vec()
}

/// Joins clips with `gap` silent samples between each consecutive pair.
pub fn concat_clips(clips: &[&[f32]], gap: usize) -> Vec<f32> {
    let total: usize = clips.iter().map(|c| c.len()).sum();
    let gaps = clips.len().saturating_sub(1) * gap;
    let mut out = Vec::with_capacity(total + gaps);
    for (i, clip) in clips.iter().enumerate() {
        if i > 0 {
            out.extend(std::iter::repeat_n(0.0, gap));
        }
        out.extend_from_slice(clip);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names() {
        assert_eq!(clip_file_name("Knight"), "Knight.wav");
        assert_eq!(clip_file_name("e4"), "e4.wav");
        let phrase = clip_file_name("Knight to f3!");
        assert!(phrase.starts_with("phrase-") && phrase.ends_with(".wav"));
        assert_eq!(phrase.len(), "phrase-".len() + 16 + ".wav".len());
        assert_eq!(phrase, clip_file_name("Knight to f3!"));
        assert_ne!(phrase, clip_file_name("Knight to f4!"));
    }

    #[test]
    fn trims_silence_keeping_pad() {
        let rate = 22_050;
        let mut s = vec![0.0; rate as usize];
        s.extend(std::iter::repeat_n(0.5, 1000));
        s.extend(vec![0.0; rate as usize]);
        let t = trim_silence(&s, rate);
        let pad = rate as usize * 10 / 1000;
        assert_eq!(t.len(), 1000 + 2 * pad);
        assert!(trim_silence(&vec![0.001; 500], rate).is_empty());
    }

    #[test]
    fn concat_length_is_sum_plus_gaps() {
        let a = vec![1.0; 100];
        let b = vec![2.0; 50];
        let c = vec![3.0; 25];
        let out = concat_clips(&[&a, &b, &c], 40);
        assert_eq!(out.len(), 100 + 50 + 25 + 2 * 40);
        assert_eq!(out[100..140], [0.0; 40]);
        assert_eq!(concat_clips(&[], 40).len(), 0);
        assert_eq!(concat_clips(&[&a], 40).len(), 100);
    }

    #[test]
    fn downmix_and_resample() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        let up = resample_linear(&[0.0, 1.0], 1, 2);
        assert_eq!(up.len(), 4);
        assert_eq!(resample_linear(&[0.25; 8], 22_050, 22_050).len(), 8);
    }
}
