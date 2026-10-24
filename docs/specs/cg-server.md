# cg-server spec

The local command center. axum 0.8 + tokio. Deps: `cg-core`, `axum`, `tokio`, `tokio-stream`, `futures-util`, `serde`, `serde_json`, `jpeg-encoder` (0.7), `shakmaty` (validation), `tracing`. Dev-deps: `reqwest` (with the `stream` feature if needed).

## API
```rust
pub struct ServerConfig { pub ports: std::ops::RangeInclusive<u16> }   // default 8000..=8010

pub struct ServerHandle { pub addr: std::net::SocketAddr, pub join: tokio::task::JoinHandle<()> }

/// Bind 127.0.0.1 on the first free port in range and serve forever.
pub async fn serve(
    cfg: ServerConfig,
    bus: cg_core::Bus,
    frames: tokio::sync::watch::Receiver<Option<cg_core::Frame>>,
    controls: tokio::sync::mpsc::Sender<cg_core::Control>,
) -> std::io::Result<ServerHandle>;
```

## Routes
| Route | Behaviour |
|---|---|
| `GET /` | `include_str!("../static/index.html")`, `text/html; charset=utf-8`. Title "kibitz". |
| `GET /overlay` | `include_str!("../static/overlay.html")`: an OBS-friendly page. |
| `GET /events` | SSE. Use `bus.subscribe()`: first `event: snapshot` with the subscription snapshot, then one `event: game` per recent event (oldest first), then live messages (`BusMsg::Snapshot` → `snapshot`, `BusMsg::Event` → `game`). Data = JSON. Keep-alive comment every **15 s** (`Sse::keep_alive`). On broadcast `Lagged`, send a fresh `snapshot` and continue. |
| `GET /frame.jpg` | 404 with body `no frame yet` until a frame exists. Otherwise JPEG (quality 75) of the latest frame, `cache-control: no-store`. **Encode at most once per frame seq**: keep a `Mutex<Option<(u64, Bytes)>>` cache shared by all requests. Do the encoding in `spawn_blocking`. |
| `GET /frame.mjpg` | `multipart/x-mixed-replace; boundary=frame` stream. Pushes the (cached) JPEG each time `frames` changes, capped at 15 fps. Encoding happens only while someone is connected. |
| `POST /control` | See below. |

## Control gate
- Content-Type must be `application/json` (parameters allowed). Otherwise **403** `forbidden`.
- If an `Origin` header is present, it must equal `http://127.0.0.1:<port>` or `http://localhost:<port>`. Otherwise **403** `forbidden`. (No Origin, e.g. curl, is allowed. This plus the JSON content type blocks cross-site form posts and simple requests.)
- Parse into `cg_core::Control`. Unknown action / missing field → **400** `{"error": "..."}`.
- Validate against the current snapshot (`bus.snapshot()`): `Move{mv}` must be legal (SAN or UCI) in `snapshot.fen` (shakmaty) → else 400 "illegal move". `NewGame{color,fen}`: color is "white"|"black", and fen (if given) parses to a legal position → else 400. `SetCorners`: all coords finite and within [-0.1, 1.1].
- Then `controls.try_send(control)`. Full → 503. OK → **202** `{"ok":true}`.

## index.html (single file, no external requests, no build step, vanilla JS)
Dark, compact, readable at a glance. Sections:
1. **Status pill** (snapshot.status_label; color by status), plus spent `$x.xx / $cap` and the model if `llm_enabled`.
2. **Buttons**: Pause/Resume (label flips with `paused`), Undo, Repeat, New game (color select + optional FEN input), Enter move (text input), Relocate board.
3. **Suggestion card** (only when `suggestion` exists): big SAN, spoken text, score_text, a confidence bar whose width is `confidence*100%`.
4. **Live view**: `<img src="/frame.mjpg">` with a `<canvas>` overlay that draws the board grid from `snapshot.corners` (board order a8,h8,h1,a1 in frame pixels; scale by displayed size / `frame_size`) and an arrow for the suggestion (from/to square centres computed via a bilinear map of the 4 corners). **Calibration mode**: "Set corners" button → user clicks the 4 board corners in on-screen order TL, TR, BR, BL → POST `set_corners` with `view` as fractions of the frame.
5. **Moves list** (numbered pairs, "no moves" placeholder).
6. **Latency strip**: detect / engine / total / frame ms and a ponder-hit badge (from `snapshot.latency`).
7. **Log**: newest first, filter tabs All / Moves / Problems (`commit`/`undo` = moves; `lost_track`, `source_missing`, `error` = problems). Frame events show changed squares, best, margin, skipped reason, vision_ms. Commit lines show "(suggested X)" when the move differs from `suggested`.
- POST with `fetch('/control', {method:'POST', headers:{'content-type':'application/json'}, body})`. Show the error text from 4xx responses inline.
- Auto-reconnect is EventSource's default; show a "disconnected" badge on error.

## overlay.html
Full-viewport `/frame.mjpg` plus the same canvas drawing (grid, suggestion arrow, status text in a corner). Transparent body background. No controls.

## Tests (`tests/web.rs`, real server on an ephemeral port range, e.g. 18000..=18050)
- `/` serves HTML containing "kibitz".
- `/frame.jpg` is 404 before a frame. After sending `Frame::from_gray(1, 64, 48, …)` into the watch, it is 200 with JPEG magic bytes `FF D8`.
- `/control`: wrong content type → 403; a foreign Origin → 403; same-origin JSON pause → 202 and the control arrives on the mpsc; illegal move → 400; unknown action → 400.
- `/events`: the first event is `snapshot`, then recent events in order, then a newly emitted event arrives.
- The port fallback works when the first port is taken.
Use `CARGO_TARGET_DIR=target/agent-server`.

## Rules
Edit only `crates/cg-server/**`. If the contract must change, report it instead.
