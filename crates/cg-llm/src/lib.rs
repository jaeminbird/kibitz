//! Optional vision-LLM client (Anthropic Messages or OpenAI chat completions).
//!
//! Used only to bootstrap or recover the board location and to break ties. It is never on the
//! per-move path. Requests are plain HTTP, since there is no official Rust SDK.

use std::{fmt, time::Duration};

use anyhow::Context as _;
use base64::Engine as _;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};

/// Default model for [`LlmConfig::new`] callers that have no preference.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";

const KEYRING_SERVICE: &str = "chessglass";
const RETRY_DELAY: Duration = Duration::from_millis(250);
const ANTHROPIC_VERSION: &str = "2023-06-01";
const LOCATE_PROMPT: &str = "This is a photo from glasses worn by a chess player looking at a physical chessboard. Return the pixel positions of the four outer corners of the 8×8 playing area (exclude any border/frame with coordinates), as fractions of image width and height, ordered top-left, top-right, bottom-right, bottom-left as they appear in the image. If no chessboard is visible set found=false.";

/// The API family a model is served by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
}

impl Provider {
    /// Provider serving `model`: `gpt-*` and `o<digit>*` are OpenAI, everything else is Anthropic.
    pub fn for_model(model: &str) -> Provider {
        let o_series = model.starts_with('o') && model.as_bytes().get(1).is_some_and(u8::is_ascii_digit);
        if model.starts_with("gpt-") || o_series {
            Provider::OpenAI
        } else {
            Provider::Anthropic
        }
    }

    fn env_var(self) -> &'static str {
        match self {
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::OpenAI => "OPENAI_API_KEY",
        }
    }

    fn key_user(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
        }
    }

    fn default_base(self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::OpenAI => "https://api.openai.com",
        }
    }
}

/// Client settings. Build with [`LlmConfig::new`] and adjust fields as needed.
#[derive(Clone)]
pub struct LlmConfig {
    pub model: String,
    pub api_key: String,
    /// Per-request timeout. Default 20 s.
    pub timeout: Duration,
    /// Retries on 429/5xx/timeout/connection errors only. Default 1.
    pub max_retries: u32,
    /// Overrides the provider's API origin (used by tests).
    pub base_url: Option<String>,
    /// USD per 1M input tokens.
    pub price_in_per_mtok: f64,
    /// USD per 1M output tokens.
    pub price_out_per_mtok: f64,
}

impl LlmConfig {
    /// Config for `model` with prices taken from [`price_for`].
    pub fn new(model: &str, api_key: String) -> Self {
        let (price_in_per_mtok, price_out_per_mtok) = price_for(model);
        Self {
            model: model.to_owned(),
            api_key,
            timeout: Duration::from_secs(20),
            max_retries: 1,
            base_url: None,
            price_in_per_mtok,
            price_out_per_mtok,
        }
    }
}

impl fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmConfig")
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Token usage and estimated USD cost of one call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Errors from [`VisionLlm`] calls.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("no API key for {0:?}")]
    NoKey(Provider),
    #[error("request timed out")]
    Timeout,
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("model refused")]
    Refused,
    #[error("unparseable answer: {0}")]
    BadAnswer(String),
    #[error(transparent)]
    Net(#[from] reqwest::Error),
}

/// Four outer corners of the playing area, TL, TR, BR, BL, as fractions of width and height.
pub type Corners = [[f32; 2]; 4];

/// Vision-LLM client for board location and move tie-breaking.
pub struct VisionLlm {
    cfg: LlmConfig,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct LocateAnswer {
    found: bool,
    #[serde(default)]
    corners: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
struct PickAnswer {
    #[serde(rename = "move")]
    mv: String,
}

impl VisionLlm {
    pub fn new(cfg: LlmConfig) -> Self {
        Self { cfg, http: reqwest::Client::new() }
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    /// Locate the playing area in a JPEG. Returns `Ok((None, cost))` if no board is visible.
    pub async fn locate_board(&self, jpeg: &[u8]) -> Result<(Option<Corners>, Cost), LlmError> {
        let schema = json!({
            "type": "object",
            "properties": {
                "found": { "type": "boolean" },
                "corners": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } }
            },
            "required": ["found", "corners"],
            "additionalProperties": false
        });
        let (ans, cost): (LocateAnswer, Cost) = self.ask(jpeg, LOCATE_PROMPT, schema).await?;
        if !ans.found {
            return Ok((None, cost));
        }
        Ok((Some(validate_corners(&ans.corners)?), cost))
    }

    /// Given the FEN before the move and a photo after it, return which candidate (SAN) was played.
    /// `Ok((None, cost))` if the model answers "none". Any other answer must be a candidate.
    pub async fn pick_move(
        &self,
        jpeg: &[u8],
        fen: &str,
        candidates: &[String],
    ) -> Result<(Option<String>, Cost), LlmError> {
        let schema = json!({
            "type": "object",
            "properties": { "move": { "type": "string" } },
            "required": ["move"],
            "additionalProperties": false
        });
        let prompt = pick_prompt(fen, candidates);
        let (ans, cost): (PickAnswer, Cost) = self.ask(jpeg, &prompt, schema).await?;
        let mv = ans.mv.trim();
        if mv.eq_ignore_ascii_case("none") {
            return Ok((None, cost));
        }
        match candidates.iter().find(|c| c.as_str() == mv) {
            Some(c) => Ok((Some(c.clone()), cost)),
            None => Err(LlmError::BadAnswer(format!("{mv:?} is not one of the candidates"))),
        }
    }

    /// Send one image + prompt with a JSON schema and parse the structured answer.
    async fn ask<T: DeserializeOwned>(
        &self,
        jpeg: &[u8],
        prompt: &str,
        schema: Value,
    ) -> Result<(T, Cost), LlmError> {
        let provider = Provider::for_model(&self.cfg.model);
        if self.cfg.api_key.is_empty() {
            return Err(LlmError::NoKey(provider));
        }
        let image = base64::engine::general_purpose::STANDARD.encode(jpeg);
        let base = self
            .cfg
            .base_url
            .as_deref()
            .unwrap_or(provider.default_base())
            .trim_end_matches('/');
        let model = &self.cfg.model;
        let (url, body) = match provider {
            Provider::Anthropic => (
                format!("{base}/v1/messages"),
                anthropic_body(model, &image, prompt, schema),
            ),
            Provider::OpenAI => (
                format!("{base}/v1/chat/completions"),
                openai_body(model, &image, prompt, schema),
            ),
        };
        let reply = self.post(provider, &url, &body).await?;
        let (text, input, output) = match provider {
            Provider::Anthropic => parse_anthropic(&reply)?,
            Provider::OpenAI => parse_openai(&reply)?,
        };
        let answer = serde_json::from_str(&text)
            .map_err(|e| LlmError::BadAnswer(format!("{e}: {}", excerpt(&text))))?;
        Ok((answer, self.cost(input, output)))
    }

    /// POST with the provider's auth headers, retrying 429/5xx/timeouts/connection errors.
    async fn post(&self, provider: Provider, url: &str, body: &Value) -> Result<Value, LlmError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let last = attempt > self.cfg.max_retries;
            let req = self.http.post(url).timeout(self.cfg.timeout).json(body);
            let req = match provider {
                Provider::Anthropic => req
                    .header("x-api-key", &self.cfg.api_key)
                    .header("anthropic-version", ANTHROPIC_VERSION),
                Provider::OpenAI => req.bearer_auth(&self.cfg.api_key),
            };
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp
                            .json::<Value>()
                            .await
                            .map_err(|e| LlmError::BadAnswer(format!("response is not JSON: {e}")));
                    }
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable && !last {
                        tracing::warn!(%status, attempt, "vision LLM request failed, retrying");
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                    let body = resp.text().await.unwrap_or_default();
                    return Err(LlmError::Http { status: status.as_u16(), body });
                }
                Err(e) => {
                    let retryable = e.is_timeout() || e.is_connect();
                    if retryable && !last {
                        tracing::warn!(error = %e, attempt, "vision LLM request failed, retrying");
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                    return Err(if e.is_timeout() { LlmError::Timeout } else { LlmError::Net(e) });
                }
            }
        }
    }

    fn cost(&self, input_tokens: u64, output_tokens: u64) -> Cost {
        let usd = input_tokens as f64 * self.cfg.price_in_per_mtok / 1e6
            + output_tokens as f64 * self.cfg.price_out_per_mtok / 1e6;
        Cost { usd, input_tokens, output_tokens }
    }
}

fn anthropic_body(model: &str, image_b64: &str, prompt: &str, schema: Value) -> Value {
    json!({
        "model": model,
        "max_tokens": 1024,
        "output_config": {
            "effort": "low",
            "format": { "type": "json_schema", "schema": schema }
        },
        "messages": [{
            "role": "user",
            "content": [
                { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": image_b64 } },
                { "type": "text", "text": prompt }
            ]
        }]
    })
}

fn openai_body(model: &str, image_b64: &str, prompt: &str, schema: Value) -> Value {
    json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "image_url", "image_url": { "url": format!("data:image/jpeg;base64,{image_b64}") } },
                { "type": "text", "text": prompt }
            ]
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": { "name": "answer", "strict": true, "schema": schema }
        }
    })
}

/// Returns (answer text, input tokens, output tokens).
fn parse_anthropic(reply: &Value) -> Result<(String, u64, u64), LlmError> {
    if reply["stop_reason"] == "refusal" {
        return Err(LlmError::Refused);
    }
    let text = reply["content"]
        .as_array()
        .and_then(|blocks| blocks.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .ok_or_else(|| LlmError::BadAnswer("no text block in response".into()))?;
    let usage = &reply["usage"];
    Ok((
        text.to_owned(),
        usage["input_tokens"].as_u64().unwrap_or(0),
        usage["output_tokens"].as_u64().unwrap_or(0),
    ))
}

/// Returns (answer text, input tokens, output tokens).
fn parse_openai(reply: &Value) -> Result<(String, u64, u64), LlmError> {
    let message = &reply["choices"][0]["message"];
    if message["refusal"].as_str().is_some_and(|r| !r.is_empty()) {
        return Err(LlmError::Refused);
    }
    let text = message["content"]
        .as_str()
        .ok_or_else(|| LlmError::BadAnswer("no message content in response".into()))?;
    let usage = &reply["usage"];
    Ok((
        text.to_owned(),
        usage["prompt_tokens"].as_u64().unwrap_or(0),
        usage["completion_tokens"].as_u64().unwrap_or(0),
    ))
}

/// Checks 4 corners lie near the image and span a non-degenerate quadrilateral (area > 1%).
fn validate_corners(pts: &[[f64; 2]]) -> Result<Corners, LlmError> {
    let [a, b, c, d] = <[[f64; 2]; 4]>::try_from(pts)
        .map_err(|_| LlmError::BadAnswer(format!("expected 4 corners, got {}", pts.len())))?;
    if pts.iter().flatten().any(|v| !(-0.05..=1.05).contains(v)) {
        return Err(LlmError::BadAnswer("corner outside the image".into()));
    }
    let quad = [a, b, c, d];
    let twice_area: f64 = (0..4)
        .map(|i| {
            let (p, q) = (quad[i], quad[(i + 1) % 4]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum();
    if twice_area.abs() / 2.0 <= 0.01 {
        return Err(LlmError::BadAnswer("degenerate board quadrilateral".into()));
    }
    Ok(quad.map(|p| [p[0] as f32, p[1] as f32]))
}

fn pick_prompt(fen: &str, candidates: &[String]) -> String {
    let side = match fen.split_whitespace().nth(1) {
        Some("b") => "Black",
        _ => "White",
    };
    format!(
        "This is a photo from glasses worn by a chess player, taken after a move on the physical board. \
         Position before the move (FEN): {fen}\n\
         {side} is to move. Which of these legal moves was played: {}?\n\
         Answer with exactly one of the listed moves, or \"none\" if the photo shows none of them. \
         Reply as JSON {{\"move\": \"...\"}}.",
        candidates.join(", ")
    )
}

fn excerpt(s: &str) -> String {
    s.chars().take(200).collect()
}

/// USD per 1M tokens (input, output). Unknown models use a conservative default.
pub fn price_for(model: &str) -> (f64, f64) {
    match model {
        "claude-opus-5-5" => (4.0, 20.0),
        "claude-sonnet-5-5" => (2.0, 10.0),
        "claude-haiku-5-5" => (0.10, 0.50),
        "claude-fable-5-1" => (10.0, 50.0),
        _ => (5.0, 25.0),
    }
}

/// API key for `p`: the env var (ANTHROPIC_API_KEY / OPENAI_API_KEY) wins, else the OS keyring.
pub fn resolve_key(p: Provider) -> Option<String> {
    if let Ok(v) = std::env::var(p.env_var()) {
        let v = v.trim();
        if !v.is_empty() {
            return Some(v.to_owned());
        }
    }
    let entry = keyring::Entry::new(KEYRING_SERVICE, p.key_user()).ok()?;
    entry.get_password().ok().filter(|k| !k.is_empty())
}

/// Store `key` for `p` in the OS keyring.
pub fn store_key(p: Provider, key: &str) -> anyhow::Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, p.key_user())
        .with_context(|| format!("opening keyring entry for {p:?}"))?;
    entry
        .set_password(key)
        .with_context(|| format!("storing key for {p:?}"))
}
