//! Video file decoding through the external `ffmpeg` / `ffprobe` programs.

use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use cg_core::{CaptureError, Frame, FrameSource};

/// Decodes a video file into RGB24 frames by piping raw video out of `ffmpeg`.
///
/// The ffmpeg child is killed and reaped when the source is dropped.
pub struct VideoFileSource {
    path: PathBuf,
    width: u32,
    height: u32,
    child: Child,
    stdout: ChildStdout,
    seq: u64,
}

impl VideoFileSource {
    /// `fps` (if `Some` and > 0) resamples with `-vf fps=N`. `realtime` adds `-re`, so the
    /// clip plays at its native speed instead of decoding as fast as possible.
    /// Dimensions come from ffprobe. End of stream → `Ok(None)`.
    pub fn open(path: &Path, fps: Option<f32>, realtime: bool) -> Result<Self, CaptureError> {
        if !path.is_file() {
            return Err(CaptureError::Other(format!("no such file: {}", path.display())));
        }
        let (width, height) = probe_dimensions(path)?;

        let mut cmd = tool_command("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
        if realtime {
            cmd.arg("-re");
        }
        cmd.arg("-i").arg(path);
        if let Some(fps) = fps.filter(|f| *f > 0.0) {
            cmd.arg("-vf").arg(format!("fps={fps}"));
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());

        let mut child = cmd
            .spawn()
            .map_err(|e| CaptureError::Other(format!("failed to start ffmpeg: {e}")))?;
        let stdout = child.stdout.take().expect("stdout was piped");
        Ok(Self { path: path.to_path_buf(), width, height, child, stdout, seq: 0 })
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        let mut rgb = vec![0u8; self.width as usize * self.height as usize * 3];
        match self.stdout.read_exact(&mut rgb) {
            Ok(()) => {
                self.seq += 1;
                Ok(Some(Frame::from_rgb(self.seq, self.width, self.height, rgb)))
            }
            // A short read at the end of the pipe is the end of the stream.
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(CaptureError::Other(format!("ffmpeg read failed: {e}"))),
        }
    }

    fn describe(&self) -> String {
        format!("file:{}", self.path.display())
    }
}

impl Drop for VideoFileSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn probe_dimensions(path: &Path) -> Result<(u32, u32), CaptureError> {
    let out = tool_command("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| CaptureError::Other(format!("failed to start ffprobe: {e}")))?;
    if !out.status.success() {
        return Err(CaptureError::Other(format!("ffprobe failed on {}", path.display())));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let parsed = text.trim().split_once('x').and_then(|(w, h)| {
        Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
    });
    match parsed {
        Some((w, h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => Err(CaptureError::Other(format!("no video stream in {}", path.display()))),
    }
}

/// Command for an external tool; on Windows, without a console window.
fn tool_command(name: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(name);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn reads_every_frame_of_a_generated_clip() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("clip.mp4");
        let status = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "1"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(status.success());

        let mut src = VideoFileSource::open(&clip, None, false).unwrap();
        let mut seqs = Vec::new();
        while let Some(f) = src.next_frame().unwrap() {
            assert_eq!((f.width, f.height), (160, 120));
            assert_eq!(f.rgb.len(), 160 * 120 * 3);
            seqs.push(f.seq);
        }
        assert_eq!(seqs, (1..=10).collect::<Vec<u64>>());
        assert!(src.next_frame().unwrap().is_none());
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(VideoFileSource::open(Path::new("does/not/exist.mp4"), None, false).is_err());
    }
}
