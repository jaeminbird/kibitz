//! Frame sources for real inputs (desktop windows, video files, image directories)
//! and the dedicated capture thread that feeds them into the pipeline.

mod images;
mod video;
mod window;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cg_core::{CaptureError, Frame, FrameSource};
use tokio::sync::{mpsc, watch};

pub use images::ImageDirSource;
pub use video::VideoFileSource;
pub use window::{list_windows, WindowSource};

const MISSING_RETRY: Duration = Duration::from_secs(1);
const BACKOFF_START: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(10);

/// Open a source from a spec string.
///
/// - `window:<matcher>` (or a bare string): live desktop window, see [`WindowSource`].
/// - `file:<path>`: video file played at native speed (`-re`) and sampled at `fps`.
/// - `dir:<path>`: image directory, looped forever, one image per `1/fps` s.
pub fn open_source(spec: &str, fps: f32) -> Result<Box<dyn FrameSource>, CaptureError> {
    if let Some(path) = spec.strip_prefix("file:") {
        Ok(Box::new(VideoFileSource::open(Path::new(path), Some(fps), true)?))
    } else if let Some(dir) = spec.strip_prefix("dir:") {
        Ok(Box::new(ImageDirSource::open(Path::new(dir), fps, true)?))
    } else {
        let matcher = spec.strip_prefix("window:").unwrap_or(spec);
        if matcher.trim().is_empty() {
            return Err(CaptureError::Other("empty window matcher".into()));
        }
        Ok(Box::new(WindowSource::new(matcher, fps)))
    }
}

/// Health of the capture thread. Sent only on transitions.
#[derive(Clone, Debug, PartialEq)]
pub enum CaptureStatus {
    Running,
    /// Source not visible right now (e.g. window closed); retrying every second.
    Missing(String),
    /// Hard failure; retrying with exponential backoff.
    Error(String),
    /// Source exhausted; the thread has exited.
    Ended,
}

/// Run `source` on a dedicated OS thread named "capture".
///
/// - Each frame is published with `frames.send_replace` (latest wins; nothing queues).
/// - Status changes are sent on `status` only when they differ from the last one sent.
/// - `SourceMissing` → `Missing`, sleep 1 s, retry. Other errors → `Error`, backoff 0.5 s
///   doubling to 10 s, reset on success. `Ok(None)` → `Ended`, thread exits.
/// - Exits within about one frame period once `stop` is set.
pub fn spawn_capture(
    mut source: Box<dyn FrameSource>,
    frames: watch::Sender<Option<Frame>>,
    status: mpsc::UnboundedSender<CaptureStatus>,
    stop: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("capture".into())
        .spawn(move || {
            let mut last: Option<CaptureStatus> = None;
            let mut backoff = BACKOFF_START;
            while !stop.load(Ordering::Relaxed) {
                match source.next_frame() {
                    Ok(Some(frame)) => {
                        backoff = BACKOFF_START;
                        transition(&mut last, &status, CaptureStatus::Running);
                        frames.send_replace(Some(frame));
                    }
                    Ok(None) => {
                        transition(&mut last, &status, CaptureStatus::Ended);
                        break;
                    }
                    Err(CaptureError::SourceMissing(msg)) => {
                        transition(&mut last, &status, CaptureStatus::Missing(msg));
                        sleep_unless_stopped(MISSING_RETRY, &stop);
                    }
                    Err(CaptureError::Other(msg)) => {
                        transition(&mut last, &status, CaptureStatus::Error(msg));
                        sleep_unless_stopped(backoff, &stop);
                        backoff = (backoff * 2).min(BACKOFF_MAX);
                    }
                }
            }
            tracing::debug!("capture thread exiting");
        })
        .expect("failed to spawn capture thread")
}

fn transition(
    last: &mut Option<CaptureStatus>,
    tx: &mpsc::UnboundedSender<CaptureStatus>,
    new: CaptureStatus,
) {
    if last.as_ref() != Some(&new) {
        let _ = tx.send(new.clone());
        *last = Some(new);
    }
}

fn sleep_unless_stopped(total: Duration, stop: &AtomicBool) {
    let deadline = Instant::now() + total;
    while !stop.load(Ordering::Relaxed) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        thread::sleep((deadline - now).min(Duration::from_millis(50)));
    }
}

/// Frame period for `fps`, or `None` when `fps <= 0` (no pacing).
fn period(fps: f32) -> Option<Duration> {
    (fps > 0.0).then(|| Duration::from_secs_f32(1.0 / fps))
}

/// Sleep only the remainder of `period` since `last`, then mark `now` as the new frame start.
fn pace(period: Option<Duration>, last: &mut Option<Instant>) {
    if let (Some(period), Some(prev)) = (period, *last) {
        let elapsed = prev.elapsed();
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
    *last = Some(Instant::now());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Yields a fixed script of results; `Ok(None)` once exhausted.
    struct Scripted(VecDeque<Result<Option<Frame>, CaptureError>>);

    impl FrameSource for Scripted {
        fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
            self.0.pop_front().unwrap_or(Ok(None))
        }
        fn describe(&self) -> String {
            "scripted".into()
        }
    }

    /// Never ends, as fast as possible.
    struct Endless;

    impl FrameSource for Endless {
        fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
            Ok(Some(Frame::from_gray(1, 1, 1, vec![0])))
        }
        fn describe(&self) -> String {
            "endless".into()
        }
    }

    fn frame(seq: u64) -> Result<Option<Frame>, CaptureError> {
        Ok(Some(Frame::from_gray(seq, 2, 2, vec![seq as u8; 4])))
    }

    #[test]
    fn spawn_capture_reports_transitions_and_keeps_last_frame() {
        let missing = || Err(CaptureError::SourceMissing("gone".into()));
        let script = vec![frame(1), frame(2), frame(3), missing(), missing(), frame(4), frame(5), Ok(None)];
        let (frames_tx, frames_rx) = watch::channel::<Option<Frame>>(None);
        let (status_tx, mut status_rx) = mpsc::unbounded_channel();
        let stop = Arc::new(AtomicBool::new(false));

        let handle = spawn_capture(Box::new(Scripted(script.into())), frames_tx, status_tx, stop);
        handle.join().unwrap();

        let mut seen = Vec::new();
        while let Ok(s) = status_rx.try_recv() {
            seen.push(s);
        }
        assert_eq!(
            seen,
            vec![
                CaptureStatus::Running,
                CaptureStatus::Missing("gone".into()),
                CaptureStatus::Running,
                CaptureStatus::Ended,
            ]
        );
        assert_eq!(frames_rx.borrow().as_ref().map(|f| f.seq), Some(5));
    }

    #[test]
    fn stop_flag_ends_thread_promptly() {
        let (frames_tx, _frames_rx) = watch::channel::<Option<Frame>>(None);
        let (status_tx, _status_rx) = mpsc::unbounded_channel();
        let stop = Arc::new(AtomicBool::new(false));

        let handle = spawn_capture(Box::new(Endless), frames_tx, status_tx, stop.clone());
        thread::sleep(Duration::from_millis(20));
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
    }

    #[test]
    fn open_source_rejects_empty_window_matcher() {
        assert!(open_source("window:  ", 5.0).is_err());
    }

    #[test]
    fn list_windows_does_not_panic() {
        let _ = list_windows();
    }
}
