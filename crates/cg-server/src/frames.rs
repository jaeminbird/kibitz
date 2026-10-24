//! JPEG encoding of the latest frame, shared by every viewer.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use cg_core::Frame;
use futures_util::{stream, Stream};
use jpeg_encoder::{ColorType, Encoder};
use tokio::sync::{watch, Mutex};

const QUALITY: u8 = 75;
/// Upper bound on MJPEG parts per second.
const MAX_FPS: u64 = 15;

/// Holds the last encoded frame. The lock is held across the encode, so each
/// frame sequence number is encoded at most once no matter how many viewers ask.
#[derive(Default)]
pub struct JpegCache(Mutex<Option<(u64, Bytes)>>);

impl JpegCache {
    /// JPEG bytes for `frame`, or for a newer cached frame if one already exists.
    pub async fn get(&self, frame: Frame) -> Result<Bytes, String> {
        let mut slot = self.0.lock().await;
        if let Some((seq, jpg)) = slot.as_ref() {
            if *seq >= frame.seq {
                return Ok(jpg.clone());
            }
        }
        let seq = frame.seq;
        let jpg = tokio::task::spawn_blocking(move || encode(&frame))
            .await
            .map_err(|e| e.to_string())??;
        *slot = Some((seq, jpg.clone()));
        Ok(jpg)
    }
}

fn encode(frame: &Frame) -> Result<Bytes, String> {
    let w = u16::try_from(frame.width).map_err(|_| "frame too wide".to_string())?;
    let h = u16::try_from(frame.height).map_err(|_| "frame too tall".to_string())?;
    let mut out = Vec::new();
    Encoder::new(&mut out, QUALITY)
        .encode(&frame.rgb, w, h, ColorType::Rgb)
        .map_err(|e| e.to_string())?;
    Ok(Bytes::from(out))
}

/// One `multipart/x-mixed-replace` part holding a JPEG.
fn part(jpg: &[u8]) -> Bytes {
    let head = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpg.len());
    let mut v = Vec::with_capacity(head.len() + jpg.len() + 2);
    v.extend_from_slice(head.as_bytes());
    v.extend_from_slice(jpg);
    v.extend_from_slice(b"\r\n");
    Bytes::from(v)
}

/// Yields the latest frame each time it changes, paced to `MAX_FPS`. Encoding
/// only happens while the stream is being polled, i.e. while someone watches.
/// Ends when the capture side goes away.
pub fn mjpeg_stream(
    frames: watch::Receiver<Option<Frame>>,
    cache: Arc<JpegCache>,
) -> impl Stream<Item = Bytes> + Send + 'static {
    let interval = Duration::from_millis(1000 / MAX_FPS);
    stream::unfold((frames, cache, 0u64), move |(mut rx, cache, last)| async move {
        let frame = loop {
            // Copy out before awaiting so no borrow is held across a suspension point.
            let cur = rx.borrow_and_update().clone();
            if let Some(f) = cur {
                if f.seq > last {
                    break f;
                }
            }
            if rx.changed().await.is_err() {
                return None;
            }
        };
        let seq = frame.seq;
        let jpg = match cache.get(frame).await {
            Ok(j) => j,
            Err(e) => {
                tracing::error!("mjpeg encode failed: {e}");
                return None;
            }
        };
        tokio::time::sleep(interval).await;
        Some((part(&jpg), (rx, cache, seq)))
    })
}
