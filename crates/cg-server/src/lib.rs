//! Local web "command center": status, controls, live view and event log.
//!
//! Binds 127.0.0.1 only. Routes: the single-page UI (`/`, `/overlay`), an SSE
//! stream of snapshots and game events (`/events`), JPEG and MJPEG frames
//! (`/frame.jpg`, `/frame.mjpg`) and a gated JSON control endpoint (`/control`).
//! Every route rejects a `Host` other than this server's own address, so a
//! DNS-rebinding page cannot read frames or events.

mod control;
mod frames;

use std::convert::Infallible;
use std::future::ready;
use std::io;
use std::net::SocketAddr;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use cg_core::{Bus, BusMsg, Control, Frame, GameEvent, Snapshot, Subscription};
use futures_util::{stream, Stream, StreamExt};
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, watch};

use crate::frames::JpegCache;

const INDEX_HTML: &str = include_str!("../static/index.html");
const OVERLAY_HTML: &str = include_str!("../static/overlay.html");

/// Ports to try, in order. The first one that binds is used.
pub struct ServerConfig {
    pub ports: RangeInclusive<u16>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig { ports: 8000..=8010 }
    }
}

/// The address the server bound to, and its serving task.
pub struct ServerHandle {
    pub addr: SocketAddr,
    pub join: tokio::task::JoinHandle<()>,
}

/// Shared by every handler.
#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) bus: Bus,
    pub(crate) frames: watch::Receiver<Option<Frame>>,
    pub(crate) controls: mpsc::Sender<Control>,
    pub(crate) jpeg: Arc<JpegCache>,
    pub(crate) port: u16,
}

/// Bind 127.0.0.1 on the first free port in range and serve forever.
pub async fn serve(
    cfg: ServerConfig,
    bus: Bus,
    frames: watch::Receiver<Option<Frame>>,
    controls: mpsc::Sender<Control>,
) -> io::Result<ServerHandle> {
    let mut last_err = None;
    for port in cfg.ports.clone() {
        match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            Ok(listener) => {
                let addr = listener.local_addr()?;
                let state = AppState {
                    bus: bus.clone(),
                    frames: frames.clone(),
                    controls: controls.clone(),
                    jpeg: Arc::new(JpegCache::default()),
                    port: addr.port(),
                };
                let app = router(state);
                let join = tokio::spawn(async move {
                    if let Err(e) = axum::serve(listener, app).await {
                        tracing::error!("command center stopped: {e}");
                    }
                });
                tracing::info!("command center at http://127.0.0.1:{}", addr.port());
                return Ok(ServerHandle { addr, join });
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "empty port range")))
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/overlay", get(overlay))
        .route("/events", get(events))
        .route("/frame.jpg", get(frame_jpg))
        .route("/frame.mjpg", get(frame_mjpg))
        .route("/control", post(control::post))
        .layer(middleware::from_fn_with_state(state.clone(), host_gate))
        .with_state(state)
}

/// 403 unless `Host` is `127.0.0.1:<port>` or `localhost:<port>`. A
/// DNS-rebinding page reaches this socket with its own hostname in `Host`.
async fn host_gate(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let ok = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|h| host_ok(h, st.port))
        && req.uri().authority().is_none_or(|a| host_ok(a.as_str(), st.port));
    if ok {
        next.run(req).await
    } else {
        (StatusCode::FORBIDDEN, "forbidden").into_response()
    }
}

fn host_ok(host: &str, port: u16) -> bool {
    host.rsplit_once(':').is_some_and(|(name, p)| {
        p == port.to_string()
            && (name == "127.0.0.1" || name.eq_ignore_ascii_case("localhost"))
    })
}

async fn index() -> Response {
    html(INDEX_HTML)
}

async fn overlay() -> Response {
    html(OVERLAY_HTML)
}

fn html(body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], body).into_response()
}

/// Snapshot first, then the recent history, then live messages. A lagged
/// listener is resynchronised with a fresh snapshot.
async fn events(State(st): State<AppState>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let Subscription { snapshot, recent, rx } = st.bus.subscribe();
    let head = stream::once(ready(snapshot_event(&snapshot)));
    let history = stream::iter(recent.into_iter().map(|e| game_event(&e)));
    let live = stream::unfold((rx, st.bus.clone()), |(mut rx, bus)| async move {
        let ev = match rx.recv().await {
            Ok(BusMsg::Snapshot(s)) => snapshot_event(&s),
            Ok(BusMsg::Event(e)) => game_event(&e),
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("sse listener lagged by {n} messages; resyncing");
                snapshot_event(&bus.snapshot())
            }
            Err(broadcast::error::RecvError::Closed) => return None,
        };
        Some((ev, (rx, bus)))
    });
    let stream = head.chain(history).chain(live);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn snapshot_event(s: &Snapshot) -> Result<Event, Infallible> {
    Ok(Event::default().event("snapshot").data(to_json(s)))
}

fn game_event(e: &GameEvent) -> Result<Event, Infallible> {
    Ok(Event::default().event("game").data(to_json(e)))
}

fn to_json<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

fn current_frame(rx: &watch::Receiver<Option<Frame>>) -> Option<Frame> {
    rx.borrow().clone()
}

async fn frame_jpg(State(st): State<AppState>) -> Response {
    let Some(frame) = current_frame(&st.frames) else {
        return (StatusCode::NOT_FOUND, "no frame yet").into_response();
    };
    match st.jpeg.get(frame).await {
        Ok(jpg) => (
            [(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")],
            jpg,
        )
            .into_response(),
        Err(e) => {
            tracing::error!("jpeg encode failed: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "encode failed").into_response()
        }
    }
}

async fn frame_mjpg(State(st): State<AppState>) -> Response {
    let parts = frames::mjpeg_stream(st.frames.clone(), st.jpeg.clone());
    (
        [
            (header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Body::from_stream(parts.map(Ok::<_, Infallible>)),
    )
        .into_response()
}
