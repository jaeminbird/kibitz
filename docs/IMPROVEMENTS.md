# Improvements backlog

Findings from a five-area review of the codebase (2026-10-24): vision, engine/game, capture/voice, control plane, and build/release. Claims marked **verified** were checked against the code or upstream sources. Everything else comes from the reviewing agent's reading and should be confirmed before acting. File references are as of commit `b4480ec`.

Severity: **H** breaks a real game or exposes data · **M** degrades behaviour or reliability · **L** polish.

---

## 1. Fix first: bugs that break a real game

### 1.1 [H] Tracker never re-acquires after a single failed frame (**verified**)
- **Where:** `crates/cg-vision/src/track.rs:105-149`, `clear()` at `:177`.
- **What:** any failure in `track()` (global shift, warp, or checker score below `min_checker`) calls `clear()`, which drops the corners and templates. The next call returns `None` at `let prev = self.corners?`. The 1.5 s grace period in `cg-app/src/vision.rs:16,109-111` assumes retries, so it is dead code. One blurred frame, a hand over a corner, or a codec keyframe loses the board until an LLM locate or a manual click. The root cause is the spec wording "None = lost (also clears state)".
- **Fix:** on failure, keep the corners, templates, and coarse reference, and return `None` for this frame only. Widen the search on consecutive misses (escalating `search_px`, then a global search scored by `checker_score`). Clear only when `vision.rs` declares loss.
- **Test:** synthetic sequence with 1, 3, and 10 garbage frames inserted; tracking must resume.

### 1.2 [H] Resolution change drops the tracker permanently (**verified**)
- **Where:** `track.rs:106-107`.
- **What:** a frame-size change calls `clear()`. WhatsApp renegotiates resolution under bandwidth pressure, and resizing the window or moving it to a monitor with different DPI also changes the size (WGC frames are in physical pixels).
- **Fix:** rescale the stored corners and templates by the size ratio and continue. Also rescale calibration corners in `vision.rs`.

### 1.3 [H] DNS rebinding can read the command center (**verified**)
- **Where:** `crates/cg-server/src/lib.rs:96-105`. There is no `Host` header check on any route.
- **What:** a malicious page that rebinds its hostname to 127.0.0.1 can GET `/frame.jpg`, `/frame.mjpg` (camera frames of the room) and `/events` (game state and log). POST `/control` is still protected by the Origin and JSON content-type gates (`control.rs:223-239`).
- **Fix:** middleware on every route that rejects any `Host` other than `127.0.0.1:<port>` or `localhost:<port>` with 403. Add a test.

### 1.4 [H] A missed read freezes the game
- **Where:** `crates/cg-app/src/orchestrator.rs:203,238`; `vision.rs:132-139`.
- **What:** the reference board is only updated when a move is accepted. If a real move is read as "ambiguous" and the LLM is absent, busy, or over budget (promotion with an LLM configured, a dark piece on a dark square giving a single-square diff, and so on), the read is dropped and nothing changes. The next real move's diff then contains two moves, cannot be explained, and the game stays stuck.
- **Fix:** keep the pending ambiguous read. When the next settled diff arrives, try two-ply inference (all legal pairs) against the old reference. Also adopt per-square adaptive references (see 2.3). As a last resort, announce "didn't catch that move" once, so the user can enter it.

### 1.5 [H] Moves made while the board is not visible are lost
- **Where:** `vision.rs:79-83` (`forget_board`), `:128-131` (Baseline), `orchestrator.rs` Resume rebaseline.
- **What:** after the wearer looks away for more than 1.5 s, or after a pause, the next settled frame becomes the reference without any check against the game position. A move made meanwhile is never read, and later inference uses the wrong side to move.
- **Fix:** keep the last game-consistent reference across loss and pause. On re-acquire, run inference over 0, 1, or 2 plies against it before accepting a new baseline.

### 1.6 [H] Stockfish stalls while pondering because stdout is not drained
- **Where:** `crates/cg-engine/src/actor.rs:186-202, 290-300`.
- **What:** stdout is only read when a request is in flight. During a long opponent think, Stockfish keeps writing `info` lines; after 3 s it also writes `currmove` lines for every root move. Once the OS pipe buffer fills, Stockfish blocks on write and stops searching, so the pondered answer is weaker than it appears.
- **Fix:** a dedicated reader task that forwards lines over a channel. While pondering, keep only the latest score and depth.

### 1.7 [M] "Lost track" counter never decays (**verified**)
- **Where:** `orchestrator.rs:232-237`; constant at `:25` says "in a row".
- **What:** `unexplained` resets only on accept, baseline, or corners. Three unrelated misreads spread over an hour make the app say "Lost track of the board" aloud.
- **Fix:** reset on "no change" and on accepted reads; count only consecutive unexplained reads.

### 1.8 [M] Undo cannot cancel an in-flight search, and the stale reply survives
- **Where:** `actor.rs:120-168`; `orchestrator.rs:290-293, 307-327, 500-536`.
- **What:**
  - `engine.stop()` queues behind a running search, so a correction can take up to `movetime + slack` (2.3 s), plus a retry.
  - `ponder`, `stop` and `new_game` are fired from `tokio::spawn`, so their order relative to `best_move` is not guaranteed.
  - `last_reply` is not cleared on Undo or NewGame.
- **Fix:**
  - Make `read_search` select on a cancel channel and send `stop` when the epoch changes.
  - Send engine commands directly on the actor channel, in order, instead of spawning.
  - Clear `last_reply` on Undo and NewGame.

---

## 2. Robustness on real footage

All testing so far uses synthetic renders: flat discs, no lean, shadow, glare or codec artifacts. Expect these to matter once real WhatsApp video is used.

### 2.1 [H] Stability gate depends on absolute brightness
- **Where:** `cg-vision/src/gate.rs:94-98`.
- **What:** auto-exposure hunting or LED flicker keeps the scene "Moving", so nothing ever settles.
- **Fix:** normalise each thumbnail by its mean (or mean and std) before differencing, and require K cells over `local_threshold` instead of one.

### 2.2 [M] Global per-board normalisation
- **Where:** `features.rs:21-35`.
- **What:** a hand, a shadow, or the zero-filled out-of-frame area shifts the mean and std for all 64 squares at once. `min_change = 0.2` was tuned on synthetic contrast (σ ≈ 50). A low-contrast vinyl board (σ ≈ 15–25) magnifies compression noise 2–3×.
- **Fix:** normalise each square locally (flat-field), or set thresholds relative to a noise floor measured per session.

### 2.3 [M] Single-shot reference
- **What:** one hand lingering in the reference frame contaminates every later read.
- **Fix:** keep a per-square reference and refresh the squares that were unchanged on each settled read.

### 2.4 [M] Tall pieces ghost into the square behind them
- **What:** from a seated, oblique head view, a piece's top projects into the square behind it, so moves add residual and get rejected as "unexplained change".
- **Fix:** ignore residual on squares directly behind (in image space) the changed squares, or learn a ghost mask per session. A piece classifier (4.1) solves this properly.

### 2.5 [M] Single-square acceptance through margin arithmetic
- **Where:** `infer.rs:109,139`.
- **What:** a move can be accepted when only one of its squares exceeded the threshold.
- **Fix:** require coverage: every square in the candidate's changed set must exceed `t`.

### 2.6 [M] Tracker drift and range
- **Where:** `track.rs:42,137,219-221,457`.
- **What:** a corner whose NCC falls below `MIN_NCC` coasts on its prediction and learns velocity from it. Templates handle rotation but not perspective or scale. Head motion faster than ±64 px/frame (~1000 px/s at 15 fps) escapes the search.
- **Fix:** see 1.1 (widening search). Consider pyramidal Lucas-Kanade (4.2).

### 2.7 [M] Promotion without an LLM assumes a queen
- **Where:** `orchestrator.rs:199-211`.
- **What:** the engine then reasons about a piece that is not on the board.
- **Fix:** a two-class appearance check on the promotion square (queen vs knight are the common cases), or announce "promotion — which piece?" and wait for a control or move entry.

### 2.8 [L] Other vision details
- Gate `local_threshold` of 20 may be within reach of sub-pixel tracking jitter on checker edges. Measure on real footage.
- Possible checker-period aliasing at 480p with small boards (`track.rs:42,192`).
- Bilinear warp without area averaging adds noise on large boards (`geometry.rs`).
- **Verified correct:** en passant and castling changed-square sets, promotion grouping, square orientation.

---

## 3. Other reliability, security and performance items

### Control plane
| Sev | Item | Where | Fix |
|---|---|---|---|
| M | API keys passed as `--key-anthropic`/`--key-openai` end up in shell history and the process list | `cg-app/src/main.rs:41-43` | Read from a prompt, stdin, or an env var |
| M | Concurrent PGN saves race on one `game.pgn.tmp` | `orchestrator.rs:646-649` | One writer task fed by a `watch` channel |
| M | LLM spend is counted only on success; a billed call that times out or retries is missed | `orchestrator.rs:440,454`; `cg-llm/src/lib.rs:283` | Estimate and charge on error paths |
| L-M | Frame channel closing is silent (status freezes) | `orchestrator.rs:139` | Emit an error and set a status |
| L-M | Shutdown: `ctrl_c()` is re-created per loop iteration; a second Ctrl-C has no listener during shutdown; the speech thread is not joined; `Engine::shutdown()` is never called | `orchestrator.rs:145`, `main.rs:210` | One signal task feeding a `CancellationToken`; bounded joins; explicit engine shutdown |
| L | `SetCorners` accepts a degenerate quad (the LLM path validates; the manual path does not) | `cg-server/src/control.rs:261-268` | Reuse `validate_corners` |
| L | Keyring read errors are swallowed as "no key" | `cg-llm/src/lib.rs:424` | Log the error. (Persistence itself works: keyring 4's default `v1` mode picks the OS store automatically. **Verified.**) |
| L | Event log, `frames/` and PGN writes grow without bound; the bus mutex is held during disk writes | `cg-core/src/bus.rs:76-79`; `orchestrator.rs:652` | `tracing-appender` non-blocking writer, rotation, and a frame cap |
| L | Snapshot published every 250 ms with the full move list, serialized once per SSE client | `orchestrator.rs:583-592`; `cg-server/src/lib.rs:127` | Publish on change only; serialize once in the bus |
| L | SSE lag drops events silently | `cg-server/src/lib.rs:129-132` | Send a `lagged` notice or replay history |
| L | No cap on MJPEG viewers | `cg-server/src/frames.rs:350-357` | Cap at about 4 concurrent streams |
| L | `Frame::from_rgba` copies about 2.8 MB twice per frame, including an RGB copy that is only needed for previews | `cg-core/src/frame.rs` | Keep RGBA plus gray and convert lazily; measure first |
| L | Persistent engine errors (`BadPosition`, `NotInstalled`) retry every second with an error event each time | `orchestrator.rs:323-372` | Back off, or give up after N attempts |
| L | `outcome()` builds n+1 EPD strings per call, several times per frame | `cg-app/src/game.rs:127-130` | shakmaty `zobrist_hash` per position, or cache in `play`/`undo` |
| L | Start FEN is sent to Stockfish verbatim | `game.rs:35-45` | Normalise with `Fen::from_position` |

### Engine install
| Sev | Item | Where | Fix |
|---|---|---|---|
| L-M | Download has no timeout | `cg-engine/src/install.rs:130` | Client timeout |
| L-M | On Windows, renaming over `stockfish.exe` fails while the engine runs | `install.rs:162` | Install under a new name and switch over; clear message |
| L | Partial `.part` file left behind on network error | `install.rs:135-145` | Clean up on error |
| L | Linux Stockfish asset only for x86_64 | `install.rs:55` | Add aarch64, or document it |
- **Verified OK:** all three SHA-256 pins match. The macOS universal build covers Apple Silicon. Files written by reqwest don't get `com.apple.quarantine`.

### Capture and voice
| Sev | Item | Where | Fix |
|---|---|---|---|
| H | A new WGC session per frame: about 13 ms per frame, and up to 3 s blocked if a session stalls | `cg-capture/src/window.rs`; xcap `windows/wgc.rs` | Persistent session (`windows-capture`, see 4.4) |
| H | macOS: xcap 0.9 uses `CGWindowListCreateImage` (deprecated in macOS 15). Without Screen Recording permission it probably returns blank frames rather than an error | xcap `macos/*` | ScreenCaptureKit (`screencapturekit` crate); `doctor` checks TCC and fails loudly |
| H | Window matching takes the largest window whose title contains the matcher, so a "WhatsApp Web" browser tab can win | `window.rs:69-81` | Let the user pin a window ID chosen in `doctor`; log the chosen window |
| M | Truncated WAV after an interrupted synthesis is never regenerated | `cg-voice/src/synth.rs:163-181`; `lib.rs:113-116` | Write to `.tmp` then rename; validate on load |
| M | Clip cache is keyed by word only, not by voice or engine | `cg-voice/src/lib.rs:113` | Put a hash of (engine, voice, rate) in the cache directory name |
| M | Unknown phrases are synthesized on the playback thread and block later `say` calls (currently latent) | `lib.rs:164,208-213` | Synthesize on a worker thread |
| M | Latency claims stop at "audio queued"; the cable, WhatsApp's jitter buffer and Bluetooth are not measured | `ARCHITECTURE.md` | Measure end to end: film a clock and record the call audio |
| M | WhatsApp noise suppression or AGC may attenuate synthetic speech (unverified) | — | Test; document the call audio settings |
| M | PowerShell is spawned without `CREATE_NO_WINDOW` (console flash); AppLocker or Constrained Language Mode would break System.Speech | `synth.rs:270-275` | Add the creation flag; `doctor` checks |
| L | DPI or resize invalidates pixel-space calibration | — | See 1.2 |
| L | ffmpeg stderr discarded; rotation metadata ignored for phone clips | `cg-capture/src/video.rs` | Capture stderr; use display dimensions and autorotate |
| L | Wayland goes through the portal and likely prompts on every session (unverified) | xcap `linux/*` | Document |

---

## 4. Strategic opportunities

### 4.1 Per-square piece classifier (`ort` / ONNX Runtime)
- Replace (or back up) the diff with a small CNN over 64 crops of 32–64 px, classifying empty, 6 roles, and colour. Run it only on settled reads, a few ms on CPU.
- Fixes or mitigates 1.4, 2.2, 2.4, 2.7, hands, and lighting together, and enables reading an arbitrary position (start mid-game with no FEN).
- **Crate:** `ort` 2.0.0-rc.13 (wraps ONNX Runtime 1.28; still a release candidate, **verified** on docs.rs).
- **Starting points (licences and viewpoint fit unverified):**
  - `dimasikson/chess-board-detector`: OpenCV corners plus a per-square CNN, useful as a reference design.
  - Roboflow Universe chess-piece datasets.
  - `dopaul/chess-corner-detection-training` on Hugging Face (YOLO corner model).
- **Plan:** collect labelled crops from recorded real games (`--run-dir` already saves frames), fine-tune, export to ONNX.

### 4.2 OpenCV tracking and corner detection (`opencv` crate)
- `calcOpticalFlowPyrLK` for the tracker and `findChessboardCornersSB` for detection. The ArUco module gives instant, robust corners if the user accepts four stickers on the board frame (an optional mode).
- **Cost:** needs a system OpenCV 4.x install, which is painful on Windows and macOS, and adds tens of MB. Prototype for the tracker only.

### 4.3 Drop WhatsApp: Meta Wearables Device Access Toolkit (**verified to exist**)
- Official SDKs:
  - https://github.com/facebook/meta-wearables-dat-ios
  - https://github.com/facebook/meta-wearables-dat-android
  - Both active (pushed 2026-10-09).
  - Docs: https://developers.meta.com/wearables/
- **Exposes:** a camera stream from Ray-Ban Meta Gen 1 and 2, Ray-Ban Display, and Oakley Meta HSTN and Vanguard, up to 720p30 (lower when Bluetooth bandwidth is limited); photo capture; experimental microphone audio. Includes a Mock Device Kit for testing without glasses.
- **Unverified:**
  - whether there is an audio-out API (the glasses are a Bluetooth headset, so phone audio routing is the likely path);
  - 1.0 versus developer-preview status (the FAQ says 1.0 began rolling out 2026-09-30, while the GitHub pages still say preview);
  - publishing limits (select partners first);
  - **whether this use case is allowed under Meta's Developer Acceptable Use Policy.** Check this before building.
- **Proposed architecture:**
  - A small companion phone app (Swift/Kotlin, using the toolkit) captures frames and sends them to the PC over the LAN, as JPEG over WebSocket first, or H.264 over WebRTC (`str0m` / `webrtc-rs`) if bandwidth requires.
  - Add a `NetworkFrameSource` behind the existing `FrameSource` trait.
  - Add a `Speaker` that sends phrase IDs back to the phone, which plays cached clips to the glasses.
  - Keep WhatsApp as a fallback.
- **Removes:** the WhatsApp call, the virtual audio cable, the call's jitter buffer, and screen-capture permissions.

### 4.4 Capture backends
| Crate | Version (crates.io, 2026-10-24) | Use |
|---|---|---|
| `windows-capture` | 2.0.1 | Persistent WGC session with frame callback (about 1 ms/frame instead of about 13 ms) |
| `screencapturekit` | 11.1.0 | macOS ScreenCaptureKit stream |
| `xcap` (current) | 0.9.8 | Keep for window enumeration and `doctor` |
| `scap` | 0.0.8 (stale since 2025-08) | Skip |

### 4.5 Engine strength and latency
- **MultiPV pre-computation:** while pondering, also search the opponent's top 2–3 replies, not only the single predicted one. The parser already handles `multipv`.
- **`UCI_LimitStrength` / `UCI_Elo`:** a "friendly strength" option.
- **Syzygy** through Stockfish's own `SyzygyPath` option, as an optional download (3–5 piece tables are about 1 GB).
- **Skip:** Polyglot books (ponder already covers opening latency), `vampirc-uci` (stale), the `stockfish` crate, pure-Rust engines (weaker, and no benefit over UCI).

### 4.6 Voice
- **`piper-rs`** 0.2.0: neural TTS through ONNX, giving one natural voice on every platform and removing PowerShell and `say`. Latency is irrelevant because clips are pre-rendered. Model files are tens of MB.
- `sherpa-onnx` 2.0.0 is an alternative (TTS support in that crate version is unverified).
- `cpal` directly only if the end-to-end measurement shows rodio buffering matters.

### 4.7 Control plane tools
- **Adopt:** `tracing-appender` (non-blocking event log).
- **Prototype only if measured:** a faster JPEG encoder.
- **Skip for now:** `tower-http` (the Host check is about 10 lines), `mimalloc`, WebSockets, config-file crates, LLM prompt caching (the prefix is too small and changes every call).

---

## 5. Build, test, release

### Findings
| Sev | Item | Fix |
|---|---|---|
| H | No CI; no fmt, clippy or deny gates; no `rust-version` (code needs 1.88 for `slice::as_chunks`) | Add `rust-version = "1.88"`, `[workspace.lints]` with `unsafe_code = "forbid"`, and the CI below |
| H | 25 of 65 tracked files use CRLF; no `.gitattributes` | `* text=auto eol=lf`, then `git add --renormalize .`; `rustfmt.toml` with `newline_style = "Unix"` |
| M-H | Default `cargo test` touches real hardware: audio devices (`cg-voice/src/lib.rs:264-285`) and windows (`cg-capture/src/lib.rs:216`). The ffmpeg test passes silently when ffmpeg is missing (`video.rs:128-130`) | Gate behind `KIBITZ_HW_TESTS=1` / `#[ignore]`; fail loudly when the gate is set |
| M | Timing-based tests: engine `elapsed` bounds and sleeps, an unbounded join in `stop_flag_ends_thread_promptly`, fixed port 18100 in `web.rs:209` (the spec says the ephemeral range) | nextest `slow-timeout`; bind port 0; poll instead of sleeping |
| M | `features.rs:166-178` is a benchmark disguised as a test | Move it to criterion |
| M | Orchestrator (683 lines) has no tests | Harness below |
| M | Dependency weight: 285 crates for the app, 463 in the lockfile. reqwest's rustls pulls aws-lc-sys, which needs NASM/cmake on Windows | Switch reqwest to `native-tls` (lowest risk); rodio `default-features = false, features = ["playback", "wav"]` |
| M-L | Nothing optional | Features `voice` and `llm` on cg-app (default on); trim tokio `full` |
| M-L | The `fake-uci` test helper ships in release builds | Move it into the cg-app integration tests or gate it with `required-features` |
| L | `panic = "abort"` is undocumented; no `strip` | Document the trade-off; add `strip = "symbols"` and measure (current exe is 9.8 MB) |
| L | Docs drift: "single static binary" (MSVC likely needs vcruntime140.dll, unverified); "linux" while only an x86_64 Stockfish pin exists | Correct the docs |
| L | `target/` is 14 GB from `agent-*` dirs; stale `chessglass.exe` | `cargo clean`; add `*.pdb` and `.DS_Store` to `.gitignore` |

### Orchestrator test harness
- Move `fake-uci` into cg-app so integration tests can use `env!("CARGO_BIN_EXE_fake-uci")` and a real `Engine`.
- Drive the `watch<Frame>` sender with `cg_vision::synth::render` frames (the same pattern as `cg-app/src/bench.rs`).
- Use a recording `Speaker` to assert what was spoken, with `llm: None`.
- **Cases:** move read and suggested; a hand is not read as a move; promotion; engine failure spoken once; budget pause; source missing then resumed; undo during a search; look-away then re-acquire (1.5); a single garbage frame (1.1).
- Wrap every await in `tokio::time::timeout`. Don't assert on latency values.

### CI (`.github/workflows/ci.yml`)
- **fmt** (ubuntu): `dtolnay/rust-toolchain@stable` with rustfmt, then `cargo fmt --all --check`.
- **test** (matrix windows-latest / macos-latest / ubuntu-latest):
  - `actions/checkout@v7`
  - `dtolnay/rust-toolchain@stable` (clippy)
  - `Swatinem/rust-cache@v2`
  - `taiki-e/install-action@v2` (cargo-nextest)
  - `cargo clippy --workspace --all-targets --locked -- -D warnings`
  - `cargo nextest run --workspace --locked`, then `cargo test --doc`
  - Linux also needs `libasound2-dev` plus the xcap/pipewire dev packages (exact list unverified).
- **msrv:** `dtolnay/rust-toolchain@1.88`, then `cargo check --workspace --locked`.
- **deps:** `cargo deny check`, `cargo machete`.
- **hw** (manual `workflow_dispatch`, ideally a self-hosted Windows runner): `KIBITZ_HW_TESTS=1`, the ignored tests, and the real-Stockfish test after `kibitz setup`.

### Releases
- **cargo-dist** (`axodotdev/cargo-dist`; latest version unverified, the GitHub tag is v0.33.1) on `v*` tags.
  - Targets: x86_64-pc-windows-msvc, aarch64-apple-darwin, x86_64-apple-darwin, x86_64-unknown-linux-gnu.
  - Installers: shell, powershell, Homebrew (a `jaeminbird/homebrew-kibitz` tap).
- **winget:** `vedantmgoyal2009/winget-releaser@v2` (needs a PAT; the first submission is usually manual). Scoop: a hand-maintained bucket.
- **macOS signing:** `apple-actions/import-codesign-certs@v7.1.0`, `codesign --options runtime`, `notarytool`. Ship a `.dmg` or `.pkg` to staple. Screen Recording permission is tied to the signature, so a stable Developer ID matters.
- **Windows signing:** Authenticode (Azure Trusted Signing or SignPath OSS; current names unverified) to avoid SmartScreen warnings.
- **Licensing:** Stockfish is GPL-3.0. Downloading it at `setup` avoids redistribution; bundling it requires GPL compliance.

### Tooling
| Tool | Recommendation |
|---|---|
| cargo-deny 0.20.2 | Adopt (licences, advisories, sources) |
| cargo-machete 0.9.2 | Adopt (unused dependencies, works on stable) |
| cargo-nextest 0.9.149 | Adopt (isolation, timeouts, retries) |
| insta 1.49.0 | Adopt for PGN and SSE JSON snapshot tests |
| cargo-hack 0.6.45 | Adopt once features exist |
| proptest 1.12.0 | Prototype for move inference |
| criterion 0.8.2 | Prototype for micro-benches (keep `kibitz bench` for the README numbers) |
| mold / lld | Prototype for linking speed; measure first |
| cargo-udeps, sccache | Skip |

---

## 6. Open admin items

- **GitHub contributors sidebar still shows Claude.**
  - The API already lists only `jaeminbird`, and history has no co-author trailers, so this is GitHub's cached view of the first push.
  - **Fix:** run `gh auth refresh -h github.com -s delete_repo`, then delete and recreate `jaeminbird/kibitz`.
  - When recreating, restore the About text "A real-time chess companion for Meta glasses, written in Rust." and the topics chess, stockfish, rust, computer-vision, low-latency, meta-glasses, then push the clean history.
  - This also removes old commits that are still reachable by SHA.
- **Line-ending normalisation** (5) could be folded into the history rewrite at the same time, if wanted.

## 7. Suggested order

1. **Section 1 bugs** (1.1–1.8). Each fix is its own commit, with a synthetic regression test.
2. **Repo hygiene:** `.gitattributes`/LF, `rust-version`, hardware-test gating, `fake-uci` move, then CI.
3. **Real-world validation:** record a real game with `--run-dir` and a screen recording of the call. Tune section 2 against the replay (`--source file:`) and measure latency end to end.
4. **Persistent capture** (4.4) and the macOS ScreenCaptureKit path.
5. **Piece classifier** (4.1), trained on the recorded footage.
6. **Meta Wearables Device Access Toolkit companion app** (4.3), after checking the acceptable-use policy.
