//! Shared contract for every kibitz crate: frames, events, snapshot,
//! controls, and the event bus. Keep this crate dependency-light; everything
//! else builds on it.

pub mod bus;
pub mod control;
pub mod event;
pub mod frame;
pub mod snapshot;
pub mod speech;

pub use bus::{Bus, BusMsg, Subscription};
pub use control::{Control, Side};
pub use event::{EventKind, GameEvent};
pub use frame::{CaptureError, Frame, FrameSource};
pub use snapshot::{Corners, Latency, Score, Snapshot, Status, Suggestion};
pub use speech::spoken_san;

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
