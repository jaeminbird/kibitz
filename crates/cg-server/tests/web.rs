//! End-to-end tests against a real server on loopback. Each test uses its own
//! port range so they can run in parallel.

use std::ops::RangeInclusive;
use std::time::Duration;

use cg_core::{Bus, Control, EventKind, Frame};
use cg_server::{serve, ServerConfig, ServerHandle};
use reqwest::header::{CONTENT_TYPE, ORIGIN};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

struct Harness {
    base: String,
    port: u16,
    bus: Bus,
    frames: watch::Sender<Option<Frame>>,
    controls: mpsc::Receiver<Control>,
    _handle: ServerHandle,
}

async fn start(ports: RangeInclusive<u16>) -> Harness {
    let bus = Bus::new(None).unwrap();
    let (frames, frames_rx) = watch::channel(None);
    let (controls_tx, controls) = mpsc::channel(8);
    let handle = serve(ServerConfig { ports }, bus.clone(), frames_rx, controls_tx)
        .await
        .expect("server should bind");
    Harness {
        base: format!("http://{}", handle.addr),
        port: handle.addr.port(),
        bus,
        frames,
        controls,
        _handle: handle,
    }
}

fn gray_frame(seq: u64) -> Frame {
    Frame::from_gray(seq, 64, 48, vec![128; 64 * 48])
}

async fn post_control(
    client: &reqwest::Client,
    h: &Harness,
    content_type: &str,
    origin: Option<&str>,
    body: &str,
) -> reqwest::Response {
    let mut req = client
        .post(format!("{}/control", h.base))
        .header(CONTENT_TYPE, content_type)
        .body(body.to_string());
    if let Some(o) = origin {
        req = req.header(ORIGIN, o);
    }
    req.send().await.unwrap()
}

#[tokio::test]
async fn index_serves_html() {
    let h = start(18000..=18010).await;
    let resp = reqwest::get(format!("{}/", h.base)).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers()[CONTENT_TYPE].to_str().unwrap().to_string();
    assert!(ct.starts_with("text/html"), "content-type was {ct}");
    assert!(resp.text().await.unwrap().contains("chessglass"));
}

#[tokio::test]
async fn frame_jpg_is_404_then_jpeg() {
    let h = start(18020..=18030).await;
    let url = format!("{}/frame.jpg", h.base);
    let resp = reqwest::get(&url).await.unwrap();
    assert_eq!(resp.status(), 404);
    assert_eq!(resp.text().await.unwrap(), "no frame yet");

    h.frames.send(Some(gray_frame(1))).unwrap();
    let resp = reqwest::get(&url).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["cache-control"], "no-store");
    let body = resp.bytes().await.unwrap();
    assert_eq!(&body[..2], &[0xFF, 0xD8], "not a JPEG");
}

#[tokio::test]
async fn frame_mjpg_streams_parts() {
    let h = start(18040..=18050).await;
    h.frames.send(Some(gray_frame(1))).unwrap();
    let mut resp = reqwest::get(format!("{}/frame.mjpg", h.base)).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers()[CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("multipart/x-mixed-replace"));
    let mut buf = Vec::new();
    while !contains(&buf, &[0xFF, 0xD8]) || !contains(&buf, b"--frame") {
        let chunk = tokio::time::timeout(Duration::from_secs(5), resp.chunk())
            .await
            .expect("timed out waiting for mjpeg part")
            .unwrap()
            .expect("stream ended early");
        buf.extend_from_slice(&chunk);
    }
}

#[tokio::test]
async fn control_gate_and_validation() {
    let mut h = start(18060..=18070).await;
    let client = reqwest::Client::new();
    let own = format!("http://127.0.0.1:{}", h.port);
    let pause = r#"{"action":"pause"}"#;

    // Wrong content type.
    let r = post_control(&client, &h, "text/plain", None, pause).await;
    assert_eq!(r.status(), 403);

    // Foreign origin.
    let r = post_control(&client, &h, "application/json", Some("http://evil.example"), pause).await;
    assert_eq!(r.status(), 403);

    // Same-origin JSON is accepted, parameters on the content type are fine.
    let r = post_control(&client, &h, "application/json; charset=utf-8", Some(&own), pause).await;
    assert_eq!(r.status(), 202);
    assert_eq!(r.json::<serde_json::Value>().await.unwrap()["ok"], true);
    assert_eq!(h.controls.recv().await.unwrap(), Control::Pause);

    // No Origin (curl-style) is accepted too.
    let r = post_control(&client, &h, "application/json", None, pause).await;
    assert_eq!(r.status(), 202);
    assert_eq!(h.controls.recv().await.unwrap(), Control::Pause);

    // Illegal move from the start position.
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"move","mv":"e5"}"#).await;
    assert_eq!(r.status(), 400);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["error"], "illegal move");

    // Legal moves in SAN and UCI.
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"move","mv":"e4"}"#).await;
    assert_eq!(r.status(), 202);
    assert_eq!(
        h.controls.recv().await.unwrap(),
        Control::Move { mv: "e4".into() }
    );
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"move","mv":"d2d4"}"#).await;
    assert_eq!(r.status(), 202);
    h.controls.recv().await.unwrap();

    // Unknown action and missing field.
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"explode"}"#).await;
    assert_eq!(r.status(), 400);
    assert!(r.json::<serde_json::Value>().await.unwrap()["error"].is_string());
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"move"}"#).await;
    assert_eq!(r.status(), 400);

    // New game: bad colour, bad FEN, good FEN.
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"new_game","color":"red"}"#).await;
    assert_eq!(r.status(), 400);
    let r = post_control(
        &client,
        &h,
        "application/json",
        None,
        r#"{"action":"new_game","color":"black","fen":"8/8/8/8/8/8/8/8 w - - 0 1"}"#,
    )
    .await;
    assert_eq!(r.status(), 400, "no kings is not a legal position");
    let r = post_control(&client, &h, "application/json", None, r#"{"action":"new_game","color":"black"}"#).await;
    assert_eq!(r.status(), 202);
    assert_eq!(
        h.controls.recv().await.unwrap(),
        Control::NewGame { color: "black".into(), fen: None }
    );

    // Corners out of range.
    let r = post_control(
        &client,
        &h,
        "application/json",
        None,
        r#"{"action":"set_corners","view":[[0,0],[1,0],[1,1],[0,2]]}"#,
    )
    .await;
    assert_eq!(r.status(), 400);
}

#[tokio::test]
async fn events_snapshot_history_then_live() {
    let h = start(18080..=18090).await;
    h.bus.emit(EventKind::Info { message: "old".into() });

    let mut resp = reqwest::get(format!("{}/events", h.base)).await.unwrap();
    let got = read_sse(&mut resp, 2).await;
    assert_eq!(got[0].0, "snapshot");
    assert!(got[0].1.get("status").is_some());
    assert_eq!(got[1].0, "game");
    assert_eq!(got[1].1["kind"], "info");
    assert_eq!(got[1].1["message"], "old");

    h.bus.emit(EventKind::Resumed);
    let got = read_sse(&mut resp, 1).await;
    assert_eq!(got[0].0, "game");
    assert_eq!(got[0].1["kind"], "resumed");
}

#[tokio::test]
async fn falls_back_to_next_port_when_first_is_taken() {
    let blocker = TcpListener::bind("127.0.0.1:18100").await.unwrap();
    let h = start(18100..=18110).await;
    assert_ne!(h.port, 18100);
    assert_eq!(h.port, 18101);
    drop(blocker);
}

/// Reads SSE blocks until `n` events have arrived. Returns (event name, parsed data).
async fn read_sse(resp: &mut reqwest::Response, n: usize) -> Vec<(String, serde_json::Value)> {
    let mut buf = String::new();
    let mut out = Vec::new();
    while out.len() < n {
        let chunk = tokio::time::timeout(Duration::from_secs(5), resp.chunk())
            .await
            .expect("timed out waiting for sse")
            .unwrap()
            .expect("sse stream ended");
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = buf.find("\n\n") {
            let block: String = buf.drain(..end + 2).collect();
            let mut name = None;
            let mut data = String::new();
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event:") {
                    name = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("data:") {
                    data.push_str(v.trim_start());
                }
            }
            if let Some(name) = name {
                out.push((name, serde_json::from_str(&data).unwrap()));
            }
        }
    }
    out
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}
