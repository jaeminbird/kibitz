//! The event bus: a bounded in-memory history, an optional append-only JSONL
//! log, and a broadcast channel to listeners (browser pages). Snapshots go
//! through the same channel but are never stored or logged.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::event::{EventKind, GameEvent};
use crate::snapshot::Snapshot;

pub const HISTORY_LIMIT: usize = 300;
const CHANNEL_CAPACITY: usize = 1024;

#[derive(Clone, Debug)]
pub enum BusMsg {
    Event(Arc<GameEvent>),
    Snapshot(Arc<Snapshot>),
}

struct Inner {
    recent: VecDeque<Arc<GameEvent>>,
    snapshot: Arc<Snapshot>,
    log: Option<BufWriter<File>>,
}

#[derive(Clone)]
pub struct Bus {
    inner: Arc<Mutex<Inner>>,
    tx: broadcast::Sender<BusMsg>,
}

/// What a new listener gets: the latest snapshot, recent history, and a
/// receiver for everything after. Taken under one lock so nothing is missed
/// or duplicated between the history and the live stream.
pub struct Subscription {
    pub snapshot: Arc<Snapshot>,
    pub recent: Vec<Arc<GameEvent>>,
    pub rx: broadcast::Receiver<BusMsg>,
}

impl Bus {
    pub fn new(log_path: Option<&Path>) -> std::io::Result<Bus> {
        let log = match log_path {
            Some(p) => {
                if let Some(dir) = p.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                Some(BufWriter::new(OpenOptions::new().create(true).append(true).open(p)?))
            }
            None => None,
        };
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        Ok(Bus {
            inner: Arc::new(Mutex::new(Inner {
                recent: VecDeque::with_capacity(HISTORY_LIMIT),
                snapshot: Arc::new(Snapshot::default()),
                log,
            })),
            tx,
        })
    }

    /// Stamp, remember, log, and broadcast an event.
    pub fn emit(&self, kind: EventKind) -> Arc<GameEvent> {
        let ev = Arc::new(GameEvent { ts_ms: crate::now_ms(), kind });
        let mut inner = self.inner.lock().unwrap();
        if inner.recent.len() == HISTORY_LIMIT {
            inner.recent.pop_front();
        }
        inner.recent.push_back(ev.clone());
        if let Some(log) = inner.log.as_mut() {
            // Logging must never take the game down.
            if let Ok(line) = serde_json::to_string(&*ev) {
                let _ = writeln!(log, "{line}").and_then(|_| log.flush());
            }
        }
        // Send while holding the lock so subscribe() can't interleave.
        let _ = self.tx.send(BusMsg::Event(ev.clone()));
        ev
    }

    /// Broadcast a snapshot without recording it.
    pub fn publish(&self, snapshot: Snapshot) {
        let snap = Arc::new(snapshot);
        let mut inner = self.inner.lock().unwrap();
        inner.snapshot = snap.clone();
        let _ = self.tx.send(BusMsg::Snapshot(snap));
    }

    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.inner.lock().unwrap().snapshot.clone()
    }

    pub fn recent(&self) -> Vec<Arc<GameEvent>> {
        self.inner.lock().unwrap().recent.iter().cloned().collect()
    }

    pub fn subscribe(&self) -> Subscription {
        let inner = self.inner.lock().unwrap();
        Subscription {
            snapshot: inner.snapshot.clone(),
            recent: inner.recent.iter().cloned().collect(),
            rx: self.tx.subscribe(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_is_bounded_and_log_keeps_everything() {
        let dir = std::env::temp_dir().join(format!("cg-bus-{}", std::process::id()));
        let path = dir.join("events.jsonl");
        let _ = std::fs::remove_file(&path);
        let bus = Bus::new(Some(&path)).unwrap();
        for i in 0..(HISTORY_LIMIT + 5) {
            bus.emit(EventKind::Info { message: i.to_string() });
        }
        assert_eq!(bus.recent().len(), HISTORY_LIMIT);
        let lines = std::fs::read_to_string(&path).unwrap().lines().count();
        assert_eq!(lines, HISTORY_LIMIT + 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn subscriber_sees_history_then_live() {
        let bus = Bus::new(None).unwrap();
        bus.emit(EventKind::Info { message: "old".into() });
        let mut sub = bus.subscribe();
        assert_eq!(sub.recent.len(), 1);
        bus.publish(Snapshot::default());
        bus.emit(EventKind::Resumed);
        assert!(matches!(sub.rx.recv().await.unwrap(), BusMsg::Snapshot(_)));
        assert!(matches!(sub.rx.recv().await.unwrap(), BusMsg::Event(_)));
    }
}
