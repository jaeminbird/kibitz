//! Platform text-to-speech, used only to pre-render clips (never on the call path).

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Synthesize several (text, wav_path) pairs with the platform TTS, in one batch.
///
/// - macOS: `say` per item, a few at a time.
/// - Windows: one PowerShell process running System.Speech over every item,
///   with the items passed through a temp JSON file.
/// - Linux: `espeak-ng` per item.
pub fn synthesize_batch(items: &[(String, PathBuf)], voice: Option<&str>) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    for (_, path) in items {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating {}", dir.display()))?;
        }
    }
    platform::synthesize(items, voice)
}

/// Run a command to completion, turning a failure into an error with its stderr.
fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let out = cmd.output().with_context(|| format!("starting {what}"))?;
    if !out.status.success() {
        bail!(
            "{what} failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// Runs `job` for each item on scoped threads, at most `FANOUT` at a time.
#[cfg(any(target_os = "macos", all(unix, not(target_os = "macos"))))]
fn fan_out(
    items: &[(String, PathBuf)],
    job: impl Fn(&str, &std::path::Path) -> Result<()> + Sync,
) -> Result<()> {
    const FANOUT: usize = 8;
    for chunk in items.chunks(FANOUT) {
        let results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|(text, path)| {
                    let job = &job;
                    s.spawn(move || job(text, path))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| bail!("TTS thread panicked")))
                .collect()
        });
        for r in results {
            r?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;

    pub fn synthesize(items: &[(String, PathBuf)], voice: Option<&str>) -> Result<()> {
        fan_out(items, |text, path| {
            let mut cmd = Command::new("say");
            cmd.arg("-o").arg(path).arg("--data-format=LEI16@22050");
            if let Some(v) = voice {
                cmd.arg("-v").arg(v);
            }
            cmd.arg(text);
            run(&mut cmd, "say")
        })
    }
}

#[cfg(windows)]
mod platform {
    use super::*;

    /// One PowerShell process. The script is a single line with no double quotes,
    /// so it survives the Windows command-line quoting; the data comes from JSON.
    const SCRIPT: &str = "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Speech; \
        $j=Get-Content -Raw -Encoding UTF8 -LiteralPath '__JSON__' | ConvertFrom-Json; \
        $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; \
        if($j.voice){$s.SelectVoice([string]$j.voice)}; \
        $f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono); \
        foreach($i in $j.items){$s.SetOutputToWaveFile([string]$i.path,$f); $s.Speak([string]$i.text)}; \
        $s.SetOutputToNull(); $s.Dispose()";

    pub fn synthesize(items: &[(String, PathBuf)], voice: Option<&str>) -> Result<()> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let json_path = std::env::temp_dir().join(format!(
            "cg-voice-{}-{nanos}.json",
            std::process::id()
        ));
        let job = serde_json::json!({
            "voice": voice,
            "items": items
                .iter()
                .map(|(text, path)| serde_json::json!({
                    "text": text,
                    "path": path.to_string_lossy(),
                }))
                .collect::<Vec<_>>(),
        });
        std::fs::write(&json_path, serde_json::to_vec(&job)?)
            .with_context(|| format!("writing {}", json_path.display()))?;

        let script = SCRIPT.replace("__JSON__", &json_path.to_string_lossy().replace('\'', "''"));
        let result = run(
            Command::new("powershell")
                .args(["-NoProfile", "-NonInteractive", "-Command", &script]),
            "powershell (System.Speech)",
        );
        let _ = std::fs::remove_file(&json_path);
        result
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use super::*;

    pub fn synthesize(items: &[(String, PathBuf)], voice: Option<&str>) -> Result<()> {
        fan_out(items, |text, path| {
            let mut cmd = Command::new("espeak-ng");
            cmd.arg("-w").arg(path);
            if let Some(v) = voice {
                cmd.arg("-v").arg(v);
            }
            cmd.arg(text);
            run(&mut cmd, "espeak-ng")
        })
    }
}

#[cfg(not(any(target_os = "macos", windows, unix)))]
mod platform {
    use super::*;

    pub fn synthesize(_items: &[(String, PathBuf)], _voice: Option<&str>) -> Result<()> {
        bail!("no text-to-speech backend for this platform")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(windows)]
    fn synthesizes_a_batch_on_windows() {
        let dir = std::env::temp_dir().join(format!("cg-voice-test-{}", std::process::id()));
        let items = vec![
            ("Knight".to_string(), dir.join("Knight.wav")),
            ("e4".to_string(), dir.join("e4.wav")),
        ];
        let t = std::time::Instant::now();
        synthesize_batch(&items, None).expect("synthesis");
        eprintln!("synthesized 2 words in {:?}", t.elapsed());
        for (_, path) in &items {
            let len = std::fs::metadata(path).expect("wav exists").len();
            assert!(len > 44, "{} is empty", path.display());
            let clip = crate::clip::load_clip(path).expect("wav decodes");
            assert!(!clip.is_empty(), "{} trimmed to nothing", path.display());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
