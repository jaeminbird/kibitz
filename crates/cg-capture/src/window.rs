//! Live desktop window capture via xcap.

use std::time::{Duration, Instant};

use cg_core::{CaptureError, Frame, FrameSource};
use xcap::Window;

use crate::{pace, period};

/// Capture a desktop window whose app name or title contains `matcher` (case-insensitive).
///
/// The matched window is cached; windows are re-enumerated only when capture fails or
/// every `REFIND_EVERY` (enumeration costs far more than the capture itself), so the
/// source still follows the window if it is closed, reopened or another match appears. Paces to `fps` (<= 0 disables pacing).
///
/// Requirements: the window must not be minimized, or capture yields nothing useful.
/// WhatsApp Desktop's window title is "WhatsApp".
/// How often to re-check which window best matches, even while capture succeeds.
const REFIND_EVERY: Duration = Duration::from_secs(2);

pub struct WindowSource {
    matcher: String,
    cached: Option<(Window, Instant)>,
    period: Option<Duration>,
    last_start: Option<Instant>,
    seq: u64,
}

impl WindowSource {
    pub fn new(matcher: &str, fps: f32) -> Self {
        Self { matcher: matcher.to_string(), cached: None, period: period(fps), last_start: None, seq: 0 }
    }
}

impl FrameSource for WindowSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        pace(self.period, &mut self.last_start);
        let stale = self.cached.as_ref().is_none_or(|(_, at)| at.elapsed() >= REFIND_EVERY);
        if stale {
            self.cached = find_window(&self.matcher).map(|w| (w, Instant::now()));
        }
        let captured = match &self.cached {
            Some((w, _)) if !w.is_minimized().unwrap_or(true) => w.capture_image().ok(),
            _ => None,
        };
        let image = match captured {
            Some(img) => img,
            None => {
                // The cached window went away or failed: look again before giving up.
                self.cached = find_window(&self.matcher).map(|w| (w, Instant::now()));
                let (window, _) = self.cached.as_ref().ok_or_else(|| {
                    CaptureError::SourceMissing(format!("no window matching {:?}", self.matcher))
                })?;
                window
                    .capture_image()
                    .map_err(|e| CaptureError::Other(format!("window capture failed: {e}")))?
            }
        };
        self.seq += 1;
        Ok(Some(Frame::from_rgba(self.seq, image.width(), image.height(), image.as_raw())))
    }

    fn describe(&self) -> String {
        format!("window:{}", self.matcher)
    }
}

/// Largest non-minimized window whose app name or title contains `matcher`.
fn find_window(matcher: &str) -> Option<Window> {
    let needle = matcher.to_lowercase();
    Window::all()
        .ok()?
        .into_iter()
        .filter(|w| !w.is_minimized().unwrap_or(true))
        .filter(|w| {
            let app = w.app_name().unwrap_or_default();
            let title = w.title().unwrap_or_default();
            app.to_lowercase().contains(&needle) || title.to_lowercase().contains(&needle)
        })
        .max_by_key(|w| u64::from(w.width().unwrap_or(0)) * u64::from(w.height().unwrap_or(0)))
}

/// `(app_name, title, width, height)` of every non-minimized window.
pub fn list_windows() -> Vec<(String, String, u32, u32)> {
    let Ok(windows) = Window::all() else {
        return Vec::new();
    };
    windows
        .into_iter()
        .filter(|w| !w.is_minimized().unwrap_or(true))
        .filter_map(|w| {
            Some((
                w.app_name().unwrap_or_default(),
                w.title().unwrap_or_default(),
                w.width().ok()?,
                w.height().ok()?,
            ))
        })
        .collect()
}
