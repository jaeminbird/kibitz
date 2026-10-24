# cg-llm spec

An optional vision LLM, used only to **bootstrap or recover** board location and to **break ties** when local inference is ambiguous. It is not on the normal per-move path. Deps: `cg-core`, `reqwest` (0.13, rustls, json), `serde`, `serde_json`, `base64`, `tokio`, `thiserror`, `tracing`, `keyring` (4). Dev-deps: `axum`, `tokio` for a mock server.

Rust has no official Anthropic SDK, so this uses raw HTTP.

## API
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider { Anthropic, OpenAI }
impl Provider { pub fn for_model(model: &str) -> Provider }   // "gpt-" or "o<digit>" prefix → OpenAI, else Anthropic

pub struct LlmConfig {
    pub model: String,                 // default "claude-opus-5-5"
    pub api_key: String,
    pub timeout: Duration,             // default 20 s
    pub max_retries: u32,              // default 1 (retry on 429/5xx/timeout/connection only)
    pub base_url: Option<String>,      // override for tests
    pub price_in_per_mtok: f64,        // USD per 1M input tokens
    pub price_out_per_mtok: f64,
}
impl LlmConfig { pub fn new(model: &str, api_key: String) -> Self }   // fills prices from `price_for`

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost { pub usd: f64, pub input_tokens: u64, pub output_tokens: u64 }

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("no API key for {0:?}")] NoKey(Provider),
    #[error("request timed out")] Timeout,
    #[error("HTTP {status}: {body}")] Http { status: u16, body: String },
    #[error("model refused")] Refused,
    #[error("unparseable answer: {0}")] BadAnswer(String),
    #[error(transparent)] Net(#[from] reqwest::Error),
}

pub struct VisionLlm;
impl VisionLlm {
    pub fn new(cfg: LlmConfig) -> Self;
    pub fn model(&self) -> &str;
    /// Locate the four outer corners of the PLAYING AREA (not the frame/border) in the viewer's
    /// on-screen order TL, TR, BR, BL, as fractions 0..1 of width/height. `Ok((None, cost))` if no board is visible.
    pub async fn locate_board(&self, jpeg: &[u8]) -> Result<(Option<[[f32; 2]; 4]>, Cost), LlmError>;
    /// Given the position before (FEN), a photo after the opponent/you moved, and the candidate
    /// moves (SAN, all legal), return which candidate was played, or None if the photo shows none of them.
    /// The answer MUST be one of `candidates` (validate; anything else → BadAnswer).
    pub async fn pick_move(&self, jpeg: &[u8], fen: &str, candidates: &[String]) -> Result<(Option<String>, Cost), LlmError>;
}

/// USD per 1M tokens (input, output). Anthropic: claude-opus-5-5 → (4.0, 20.0), claude-sonnet-5-5 → (2.0, 10.0),
/// claude-haiku-5-5 → (0.10, 0.50), claude-fable-5-1 → (10.0, 50.0). Unknown models → (5.0, 25.0) (conservative).
pub fn price_for(model: &str) -> (f64, f64);

/// Env var (ANTHROPIC_API_KEY / OPENAI_API_KEY) wins; else keyring service "kibitz", user "anthropic"/"openai".
pub fn resolve_key(p: Provider) -> Option<String>;
pub fn store_key(p: Provider, key: &str) -> anyhow::Result<()>;
```

## Wire formats
**Anthropic** `POST {base}/v1/messages`, headers `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`. Body:
```json
{
  "model": "<model>",
  "max_tokens": 1024,
  "output_config": {
    "effort": "low",
    "format": { "type": "json_schema", "schema": { ...see below, with "additionalProperties": false and "required" } }
  },
  "messages": [{ "role": "user", "content": [
     { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "<b64>" } },
     { "type": "text", "text": "<prompt>" }
  ]}]
}
```
- Do NOT send `thinking`, `temperature`, or an assistant prefill: these return 400 on current models. `effort: "low"` keeps it fast.
- Response: check `stop_reason`. `"refusal"` → `LlmError::Refused`. Otherwise take the first `content` block with `type == "text"` and parse its `text` as JSON. Usage: `usage.input_tokens`, `usage.output_tokens`.
- Schemas: locate → `{ "found": bool, "corners": [[x,y]×4] }` (corners: array of arrays of numbers). pick → `{ "move": string }`, where the prompt says to answer with exactly one of the listed SAN moves or "none".

**OpenAI** `POST {base}/v1/chat/completions`, `Authorization: Bearer`. Body: `model`, `messages: [{role:user, content:[{type:"text",text},{type:"image_url",image_url:{url:"data:image/jpeg;base64,…"}}]}]`, `response_format: {type:"json_schema", json_schema:{name, strict:true, schema}}`. Parse `choices[0].message.content` as JSON. Usage: `usage.prompt_tokens`/`completion_tokens`.

## Prompts
Keep them short and precise, e.g. for locate: "This is a photo from glasses worn by a chess player looking at a physical chessboard. Return the pixel positions of the four outer corners of the 8×8 playing area (exclude any border/frame with coordinates), as fractions of image width and height, ordered top-left, top-right, bottom-right, bottom-left as they appear in the image. If no chessboard is visible set found=false." For pick_move, include the FEN, whose turn it is, and the candidate list.

Validate the corners: 4 points, all within [-0.05, 1.05], non-degenerate quadrilateral (area > 1% of the image). Otherwise → BadAnswer.

## Tests (mock server, no real network)
Spin up an axum server on 127.0.0.1:0 in the test that records the request JSON and returns canned responses:
- Anthropic locate: request has the image block, `output_config.format.type == "json_schema"`, and no `thinking` field. Response parsing gives corners and cost (prices × tokens).
- Anthropic `stop_reason: "refusal"` → Refused.
- pick_move answer not in the candidates → BadAnswer; "none" → Ok(None).
- OpenAI locate path parses.
- 500 then 200 → succeeds (retry once). Two 500s → Http error.
- Timeout (server sleeps longer than a 200 ms test timeout) → Timeout.
Use `CARGO_TARGET_DIR=target/agent-llm`.

## Rules
Edit only `crates/cg-llm/**`. If the contract must change, report it instead.
