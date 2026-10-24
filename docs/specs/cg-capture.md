# cg-capture spec

Implements `cg_core::FrameSource` for real inputs, plus the capture thread. Deps: `cg-core`, `xcap` (0.9, window capture on macOS/Windows/Linux), `image` (decode for the image-dir source), `tokio` (watch/mpsc only), `tracing`, `thiserror`/`anyhow`. ffmpeg/ffprobe are invoked as external programs (`std::process`), not linked.

## API
```rust
/// Capture a desktop window whose app name or title contains `matcher` (case-insensitive),
/// e.g. "WhatsApp". Picks the largest non-minimized match. Paces to `fps`.
pub struct WindowSource;
impl WindowSource { pub fn new(matcher: &str, fps: f32) -> Self }

/// Decode a video file with ffmpeg into RGB24 frames. `realtime` adds `-re` (play at native speed);
/// `fps` (if Some) adds `-vf fps=N`. Dimensions come from ffprobe. EOF → Ok(None).
pub struct VideoFileSource;
impl VideoFileSource { pub fn open(path: &Path, fps: Option<f32>, realtime: bool) -> Result<Self, CaptureError> }

/// Every .png/.jpg in a directory, sorted by name, one per `1/fps` s (no pacing if fps <= 0).
/// After the last one → Ok(None) (or loop forever if `looping`).
pub struct ImageDirSource;
impl ImageDirSource { pub fn open(dir: &Path, fps: f32, looping: bool) -> Result<Self, CaptureError> }

/// "window:WhatsApp", "file:/path/clip.mp4", "dir:/path/frames". A bare string with no prefix = window matcher.
pub fn open_source(spec: &str, fps: f32) -> Result<Box<dyn FrameSource>, CaptureError>;

/// For `doctor`: (app_name, title, width, height) of every capturable window.
pub fn list_windows() -> Vec<(String, String, u32, u32)>;

#[derive(Clone, Debug, PartialEq)]
pub enum CaptureStatus { Running, Missing(String), Error(String), Ended }

/// Run `source` on a dedicated OS thread named "capture".
/// - Each frame: `frames.send_replace(Some(frame))` (latest wins; nobody queues).
/// - Status changes (only on transitions) are sent on `status`.
/// - SourceMissing → status Missing, sleep 1 s, retry. Other errors → status Error, exponential
///   backoff 0.5 s doubling to 10 s max, reset on success. Ok(None) → status Ended, thread exits.
/// - Exits promptly (≤ one frame period) when `stop` is set.
pub fn spawn_capture(
    source: Box<dyn FrameSource>,
    frames: tokio::sync::watch::Sender<Option<cg_core::Frame>>,
    status: tokio::sync::mpsc::UnboundedSender<CaptureStatus>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<()>;
```
- Sources assign `seq` (1, 2, 3, …) themselves and build frames with `Frame::from_rgba` / `Frame::from_rgb`.
- Pacing: sleep only the remainder of the frame period (measure capture time).
- xcap 0.9's API may differ from what you remember. Read its source under `~/.cargo/registry/src/*/xcap-0.9*/` to get it right (window list, `app_name()`, `title()`, `is_minimized()`, `capture_image()` returning an `RgbaImage`; methods may return `Result`).
- Windows: WhatsApp Desktop's window title is "WhatsApp". It must not be minimized for capture to work. Document this in a doc comment.

## Tests
- `ImageDirSource`: write 3 PNGs to a temp dir (use the `image` crate), read them back in order with sizes and seqs right, then None.
- `VideoFileSource`: if `ffmpeg` is on PATH, generate a 1-second 160×120 test clip with `ffmpeg -f lavfi -i testsrc=size=160x120:rate=10 -t 1 out.mp4`, then read ~10 frames and None. Skip (return early, with a printed note) if ffmpeg is missing.
- `spawn_capture` with a fake in-test `FrameSource` that yields 3 frames, then `SourceMissing` twice, then frames, then None. Assert the status transitions and that the watch holds the last frame.
- `list_windows()` runs without panicking (no assertion on content).
- Use `CARGO_TARGET_DIR=target/agent-capture`.

## Rules
Edit only `crates/cg-capture/**`. If the contract must change, report it instead.
