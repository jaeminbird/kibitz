//! Tests against an in-process axum mock server on 127.0.0.1:0. No real network.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    response::IntoResponse,
    Json, Router,
};
use base64::Engine as _;
use cg_llm::{price_for, LlmConfig, LlmError, Provider, VisionLlm};
use serde_json::{json, Value};

const JPEG: &[u8] = b"fake-jpeg-bytes";
const FEN_START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SQUARE_CORNERS: &str = r#"{"found":true,"corners":[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]}"#;

#[derive(Clone)]
struct Canned {
    status: u16,
    body: Value,
    delay: Duration,
}

fn ok(body: Value) -> Canned {
    Canned { status: 200, body, delay: Duration::ZERO }
}

fn status(code: u16) -> Canned {
    Canned { status: code, body: json!({ "error": "boom" }), delay: Duration::ZERO }
}

#[derive(Clone)]
struct Seen {
    path: String,
    headers: HeaderMap,
    body: Value,
}

#[derive(Default)]
struct Shared {
    queue: Mutex<VecDeque<Canned>>,
    seen: Mutex<Vec<Seen>>,
}

async fn handle(
    State(s): State<Arc<Shared>>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    s.seen.lock().unwrap().push(Seen { path: uri.path().to_owned(), headers, body: parsed });
    let next = s.queue.lock().unwrap().pop_front().unwrap_or(Canned {
        status: 599,
        body: json!({ "error": "no canned response left" }),
        delay: Duration::ZERO,
    });
    if !next.delay.is_zero() {
        tokio::time::sleep(next.delay).await;
    }
    (StatusCode::from_u16(next.status).unwrap(), Json(next.body))
}

struct Mock {
    base: String,
    shared: Arc<Shared>,
}

impl Mock {
    async fn start(canned: Vec<Canned>) -> Mock {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::from(canned)),
            seen: Mutex::default(),
        });
        let app = Router::new().fallback(handle).with_state(shared.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Mock { base: format!("http://{addr}"), shared }
    }

    fn seen(&self) -> Vec<Seen> {
        self.shared.seen.lock().unwrap().clone()
    }
}

fn config(model: &str, base: &str) -> LlmConfig {
    let mut cfg = LlmConfig::new(model, "test-key".into());
    cfg.base_url = Some(base.to_owned());
    cfg
}

fn anthropic_reply(text: &str, stop_reason: &str, input: u64, output: u64) -> Value {
    json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "stop_reason": stop_reason,
        "content": [{ "type": "text", "text": text }],
        "usage": { "input_tokens": input, "output_tokens": output }
    })
}

fn openai_reply(text: &str, prompt: u64, completion: u64) -> Value {
    json!({
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text } }],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion }
    })
}

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

fn assert_corners(got: [[f32; 2]; 4], want: [[f32; 2]; 4]) {
    for (g, w) in got.iter().zip(want.iter()) {
        for (gv, wv) in g.iter().zip(w.iter()) {
            assert!((gv - wv).abs() < 1e-6, "{got:?} != {want:?}");
        }
    }
}

#[tokio::test]
async fn anthropic_locate_sends_schema_and_parses_corners() {
    let mock = Mock::start(vec![ok(anthropic_reply(SQUARE_CORNERS, "end_turn", 1000, 100))]).await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    let (corners, cost) = llm.locate_board(JPEG).await.unwrap();

    assert_corners(corners.unwrap(), [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]);
    assert_eq!(cost.input_tokens, 1000);
    assert_eq!(cost.output_tokens, 100);
    // 1000 * $4/M + 100 * $20/M
    assert_close(cost.usd, 0.006);

    let seen = mock.seen();
    assert_eq!(seen.len(), 1);
    let req = &seen[0];
    assert_eq!(req.path, "/v1/messages");
    assert_eq!(req.headers["x-api-key"], "test-key");
    assert_eq!(req.headers["anthropic-version"], "2023-06-01");
    assert_eq!(req.body["model"], "claude-opus-5-5");
    assert_eq!(req.body["output_config"]["effort"], "low");
    assert_eq!(req.body["output_config"]["format"]["type"], "json_schema");
    assert_eq!(req.body["output_config"]["format"]["schema"]["additionalProperties"], false);
    assert!(req.body.get("thinking").is_none());
    assert!(req.body.get("temperature").is_none());
    let content = &req.body["messages"][0]["content"];
    assert_eq!(content[0]["type"], "image");
    assert_eq!(content[0]["source"]["type"], "base64");
    assert_eq!(content[0]["source"]["media_type"], "image/jpeg");
    let data = content[0]["source"]["data"].as_str().unwrap();
    assert_eq!(base64::engine::general_purpose::STANDARD.decode(data).unwrap(), JPEG);
    assert_eq!(content[1]["type"], "text");
}

#[tokio::test]
async fn anthropic_locate_not_found_is_none() {
    let reply = anthropic_reply(r#"{"found":false,"corners":[]}"#, "end_turn", 10, 5);
    let mock = Mock::start(vec![ok(reply)]).await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    let (corners, _) = llm.locate_board(JPEG).await.unwrap();
    assert!(corners.is_none());
}

#[tokio::test]
async fn invalid_corners_are_bad_answers() {
    let collinear = r#"{"found":true,"corners":[[0,0],[0.5,0],[1,0],[0.5,0]]}"#;
    let off_image = r#"{"found":true,"corners":[[0.1,0.1],[1.5,0.1],[0.9,0.9],[0.1,0.9]]}"#;
    let mock = Mock::start(vec![
        ok(anthropic_reply(collinear, "end_turn", 1, 1)),
        ok(anthropic_reply(off_image, "end_turn", 1, 1)),
    ])
    .await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::BadAnswer(_))));
    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::BadAnswer(_))));
}

#[tokio::test]
async fn anthropic_refusal_is_refused() {
    let mock = Mock::start(vec![ok(anthropic_reply("", "refusal", 10, 0))]).await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::Refused)));
}

#[tokio::test]
async fn pick_move_only_accepts_candidates() {
    let mock = Mock::start(vec![
        ok(anthropic_reply(r#"{"move":"Nf3"}"#, "end_turn", 10, 3)),
        ok(anthropic_reply(r#"{"move":"none"}"#, "end_turn", 10, 3)),
        ok(anthropic_reply(r#"{"move":"e4"}"#, "end_turn", 10, 3)),
    ])
    .await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));
    let candidates = vec!["e4".to_owned(), "d4".to_owned()];

    assert!(matches!(
        llm.pick_move(JPEG, FEN_START, &candidates).await,
        Err(LlmError::BadAnswer(_))
    ));
    let (mv, _) = llm.pick_move(JPEG, FEN_START, &candidates).await.unwrap();
    assert_eq!(mv, None);
    let (mv, _) = llm.pick_move(JPEG, FEN_START, &candidates).await.unwrap();
    assert_eq!(mv.as_deref(), Some("e4"));

    let seen = mock.seen();
    let text = seen[2].body["messages"][0]["content"][1]["text"].as_str().unwrap();
    assert!(text.contains(FEN_START));
    assert!(text.contains("e4, d4"));
    assert_eq!(seen[2].body["output_config"]["format"]["schema"]["required"][0], "move");
}

#[tokio::test]
async fn openai_locate_parses() {
    let mock = Mock::start(vec![ok(openai_reply(SQUARE_CORNERS, 1000, 100))]).await;
    let llm = VisionLlm::new(config("gpt-test", &mock.base));

    let (corners, cost) = llm.locate_board(JPEG).await.unwrap();
    assert_corners(corners.unwrap(), [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]);
    assert_eq!((cost.input_tokens, cost.output_tokens), (1000, 100));
    // Unknown model: default prices $5/M in, $25/M out.
    assert_close(cost.usd, 0.005 + 0.0025);

    let seen = mock.seen();
    let req = &seen[0];
    assert_eq!(req.path, "/v1/chat/completions");
    assert_eq!(req.headers["authorization"], "Bearer test-key");
    assert_eq!(req.body["response_format"]["type"], "json_schema");
    assert_eq!(req.body["response_format"]["json_schema"]["strict"], true);
    let content = &req.body["messages"][0]["content"];
    assert_eq!(content[0]["type"], "image_url");
    assert!(content[0]["image_url"]["url"].as_str().unwrap().starts_with("data:image/jpeg;base64,"));
}

#[tokio::test]
async fn retries_once_on_500() {
    let mock = Mock::start(vec![
        status(500),
        ok(anthropic_reply(SQUARE_CORNERS, "end_turn", 1, 1)),
    ])
    .await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    let (corners, _) = llm.locate_board(JPEG).await.unwrap();
    assert!(corners.is_some());
    assert_eq!(mock.seen().len(), 2);
}

#[tokio::test]
async fn two_500s_is_http_error() {
    let mock = Mock::start(vec![status(500), status(500)]).await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    match llm.locate_board(JPEG).await {
        Err(LlmError::Http { status, .. }) => assert_eq!(status, 500),
        other => panic!("expected Http error, got {other:?}"),
    }
    assert_eq!(mock.seen().len(), 2);
}

#[tokio::test]
async fn client_error_is_not_retried() {
    let mock = Mock::start(vec![status(400), ok(anthropic_reply(SQUARE_CORNERS, "end_turn", 1, 1))]).await;
    let llm = VisionLlm::new(config("claude-opus-5-5", &mock.base));

    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::Http { status: 400, .. })));
    assert_eq!(mock.seen().len(), 1);
}

#[tokio::test]
async fn slow_server_times_out() {
    let slow = Canned {
        status: 200,
        body: anthropic_reply(SQUARE_CORNERS, "end_turn", 1, 1),
        delay: Duration::from_secs(2),
    };
    let mock = Mock::start(vec![slow]).await;
    let mut cfg = config("claude-opus-5-5", &mock.base);
    cfg.timeout = Duration::from_millis(200);
    cfg.max_retries = 0;
    let llm = VisionLlm::new(cfg);

    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::Timeout)));
}

#[tokio::test]
async fn missing_key_is_no_key() {
    let mock = Mock::start(vec![]).await;
    let mut cfg = config("claude-opus-5-5", &mock.base);
    cfg.api_key.clear();
    let llm = VisionLlm::new(cfg);

    assert!(matches!(llm.locate_board(JPEG).await, Err(LlmError::NoKey(Provider::Anthropic))));
    assert!(mock.seen().is_empty());
}

#[test]
fn provider_for_model() {
    assert_eq!(Provider::for_model("gpt-4o"), Provider::OpenAI);
    assert_eq!(Provider::for_model("o3-mini"), Provider::OpenAI);
    assert_eq!(Provider::for_model("o1"), Provider::OpenAI);
    assert_eq!(Provider::for_model("claude-opus-5-5"), Provider::Anthropic);
    assert_eq!(Provider::for_model("ollama-thing"), Provider::Anthropic);
    assert_eq!(Provider::for_model("o"), Provider::Anthropic);
}

#[test]
fn prices_and_config_defaults() {
    assert_eq!(price_for("claude-opus-5-5"), (4.0, 20.0));
    assert_eq!(price_for("claude-sonnet-5-5"), (2.0, 10.0));
    assert_eq!(price_for("claude-haiku-5-5"), (0.10, 0.50));
    assert_eq!(price_for("claude-fable-5-1"), (10.0, 50.0));
    assert_eq!(price_for("mystery"), (5.0, 25.0));

    let cfg = LlmConfig::new("claude-opus-5-5", "k".into());
    assert_eq!(cfg.model, "claude-opus-5-5");
    assert_eq!(cfg.timeout, Duration::from_secs(20));
    assert_eq!(cfg.max_retries, 1);
    assert!(cfg.base_url.is_none());
    assert_eq!((cfg.price_in_per_mtok, cfg.price_out_per_mtok), (4.0, 20.0));
    assert!(!format!("{cfg:?}").contains("\"k\""));
}
