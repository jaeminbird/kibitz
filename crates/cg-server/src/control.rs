//! `POST /control`: the same-origin gate, then parse and validate against the
//! current snapshot, then queue the command for the game loop.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use cg_core::{Control, Snapshot};
use shakmaty::fen::Fen;
use shakmaty::san::SanPlus;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess};
use tokio::sync::mpsc::error::TrySendError;

use crate::AppState;

pub async fn post(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    // A cross-site form cannot send application/json without a preflight, and a foreign Origin is refused.
    if !is_json(&headers) || !origin_ok(&headers, st.port) {
        return (StatusCode::FORBIDDEN, "forbidden").into_response();
    }
    let ctl: Control = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    if let Err(msg) = validate(&ctl, &st.bus.snapshot()) {
        return json_error(StatusCode::BAD_REQUEST, &msg);
    }
    match st.controls.try_send(ctl) {
        Ok(()) => json_response(StatusCode::ACCEPTED, r#"{"ok":true}"#.into()),
        Err(TrySendError::Full(_)) => json_error(StatusCode::SERVICE_UNAVAILABLE, "busy, try again"),
        Err(TrySendError::Closed(_)) => {
            json_error(StatusCode::SERVICE_UNAVAILABLE, "game loop is not running")
        }
    }
}

/// Content-Type must be application/json; parameters such as charset are allowed.
fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
}

/// A missing Origin (curl, scripts) is allowed; a present one must be this server.
fn origin_ok(headers: &HeaderMap, port: u16) -> bool {
    match headers.get(header::ORIGIN) {
        None => true,
        Some(v) => v.to_str().is_ok_and(|o| {
            o == format!("http://127.0.0.1:{port}") || o == format!("http://localhost:{port}")
        }),
    }
}

/// Checks a parsed command against the position it would apply to.
pub(crate) fn validate(ctl: &Control, snap: &Snapshot) -> Result<(), String> {
    match ctl {
        Control::Move { mv } => {
            let pos = position(&snap.fen)?;
            if parse_legal(mv.trim(), &pos) {
                Ok(())
            } else {
                Err("illegal move".into())
            }
        }
        Control::NewGame { color, fen } => {
            if color != "white" && color != "black" {
                return Err(r#"color must be "white" or "black""#.into());
            }
            if let Some(fen) = fen {
                position(fen)?;
            }
            Ok(())
        }
        Control::SetCorners { view } => {
            let ok = view.iter().flatten().all(|c| c.is_finite() && (-0.1..=1.1).contains(c));
            if ok {
                Ok(())
            } else {
                Err("corners must be finite and within [-0.1, 1.1]".into())
            }
        }
        Control::Pause | Control::Resume | Control::Undo | Control::Repeat | Control::Relocate => {
            Ok(())
        }
    }
}

fn position(fen: &str) -> Result<Chess, String> {
    let setup = Fen::from_ascii(fen.as_bytes()).map_err(|e| format!("invalid fen: {e}"))?;
    setup
        .into_position::<Chess>(CastlingMode::Standard)
        .map_err(|e| format!("illegal position: {e}"))
}

/// True if `text` is a SAN (with optional check suffix) or UCI move legal in `pos`.
fn parse_legal(text: &str, pos: &Chess) -> bool {
    let san = SanPlus::from_ascii(text.as_bytes())
        .ok()
        .and_then(|s| s.san.to_move(pos).ok());
    san.is_some()
        || UciMove::from_ascii(text.as_bytes())
            .ok()
            .and_then(|u| u.to_move(pos).ok())
            .is_some()
}

fn json_error(status: StatusCode, msg: &str) -> Response {
    json_response(status, serde_json::json!({ "error": msg }).to_string())
}

fn json_response(status: StatusCode, body: String) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}
