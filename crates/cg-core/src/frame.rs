use std::sync::Arc;
use std::time::Instant;

/// One captured video frame. Cheap to clone (pixel buffers are shared).
///
/// Both an RGB copy (for JPEG preview / LLM upload) and a grayscale copy
/// (for vision) are kept so neither consumer has to convert per use.
#[derive(Clone)]
pub struct Frame {
    /// Monotonic sequence number assigned by the capture thread, starting at 1.
    pub seq: u64,
    /// When the frame was captured (monotonic clock, for latency math).
    pub at: Instant,
    /// Capture time in Unix milliseconds (for display / logs).
    pub ts_ms: u64,
    pub width: u32,
    pub height: u32,
    /// Packed RGB8, `width * height * 3` bytes, row-major, no padding.
    pub rgb: Arc<[u8]>,
    /// Luma8, `width * height` bytes, row-major, no padding.
    pub gray: Arc<[u8]>,
}

impl Frame {
    /// Build a frame from packed RGBA8 (the format most capture APIs return).
    pub fn from_rgba(seq: u64, width: u32, height: u32, rgba: &[u8]) -> Frame {
        let n = (width * height) as usize;
        assert!(rgba.len() >= n * 4, "rgba buffer too small");
        let mut rgb = Vec::with_capacity(n * 3);
        let mut gray = Vec::with_capacity(n);
        for px in rgba[..n * 4].as_chunks::<4>().0 {
            rgb.extend_from_slice(&px[..3]);
            gray.push(luma(px[0], px[1], px[2]));
        }
        Frame::new(seq, width, height, rgb, gray)
    }

    /// Build a frame from packed RGB8.
    pub fn from_rgb(seq: u64, width: u32, height: u32, rgb: Vec<u8>) -> Frame {
        let n = (width * height) as usize;
        assert!(rgb.len() >= n * 3, "rgb buffer too small");
        let gray = rgb[..n * 3].as_chunks::<3>().0.iter().map(|p| luma(p[0], p[1], p[2])).collect();
        Frame::new(seq, width, height, rgb, gray)
    }

    /// Build a frame from Luma8 only (tests / synthetic input). RGB is the gray value repeated.
    pub fn from_gray(seq: u64, width: u32, height: u32, gray: Vec<u8>) -> Frame {
        let rgb = gray.iter().flat_map(|&g| [g, g, g]).collect();
        Frame::new(seq, width, height, rgb, gray)
    }

    fn new(seq: u64, width: u32, height: u32, rgb: Vec<u8>, gray: Vec<u8>) -> Frame {
        Frame {
            seq,
            at: Instant::now(),
            ts_ms: crate::now_ms(),
            width,
            height,
            rgb: rgb.into(),
            gray: gray.into(),
        }
    }
}

#[inline]
fn luma(r: u8, g: u8, b: u8) -> u8 {
    // BT.601 integer approximation.
    ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("seq", &self.seq)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// The window/device to capture from isn't there right now. Retry later.
    #[error("capture source not visible: {0}")]
    SourceMissing(String),
    /// Anything else (I/O, decode, permission). Retry with backoff.
    #[error("capture failed: {0}")]
    Other(String),
}

/// A blocking producer of frames. Implementations run on a dedicated thread.
pub trait FrameSource: Send {
    /// Block until the next frame is available.
    /// `Ok(None)` means the source is exhausted (e.g. a recording ended).
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError>;

    /// Human-readable description for logs / doctor.
    fn describe(&self) -> String;
}
