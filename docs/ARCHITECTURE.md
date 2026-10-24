# chessglass — architecture

A Rust rewrite of *chesshacker*. It does the same job: someone wearing Meta glasses video-calls a computer, and the computer reads the over-the-board game, asks Stockfish for a move, and speaks it into the call. Like the original, it is meant for friendly games where everyone at the board knows about it, not for rated or tournament play.

The goal of the rewrite is the time from **opponent's hand leaves the board → move spoken in the earpiece**, plus cost and footprint.

## Latency budget

| Stage | Original (TS) | chessglass | How |
|---|---|---|---|
| Notice a frame worth reading | poll 1s (opp) / 3s (you) | ≤ 1 frame (66 ms @ 15 fps) | Continuous capture thread + `watch` channel (latest-frame-wins, never queued) |
| "Is the board readable / hand gone?" | Jev cloud pre-check, 5 s timeout | ~0.1 ms local | Motion-energy stability gate on a 64×36 thumbnail |
| Read the move | Cloud vision LLM, 1–10 s, 45 s timeout, $$ | ~1–3 ms local | Homography warp + per-square diff scored against *legal moves only* |
| Locate the board | LLM every 5 s | ~1 ms/frame local | Frame-to-frame corner tracking (block matching); LLM or a manual click only to bootstrap or recover |
| Engine | spawn + UCI handshake per request + 500 ms think | ~0 ms on ponder hit; ≤ movetime otherwise | Persistent Stockfish, warm hash, UCI `go ponder` / `ponderhit` |
| Speak | spawn `say` per phrase | < 10 ms to first sample | Pre-rendered word clips concatenated and played directly to the output device |
| **Total (expected move)** | **~3–15 s** | **~stability window (≈300 ms) + a few ms** | |

The remaining fixed cost is the stability window: you can't read a move while a hand is still over the board. It's configurable (`--settle-ms`).

## Decision record: original → new

| # | Original decision | Problem | New decision |
|---|---|---|---|
| 1 | Node/TypeScript single process, `sharp` + `ffmpeg-static` | ~100 MB+ install, GC pauses, native addon friction | Single static Rust binary (~10 MB), no runtime |
| 2 | Vision LLM reads the full board every move | Slow, costs money per move, non-deterministic, frames leave the machine | **Local move inference.** The tracker already knows the position, so the camera only has to tell *which of ~30 legal moves* happened. Diff warped squares vs the reference taken at the last commit; score each legal move by how well its changed-square set (from, to, castling rook, en-passant victim) explains the diff. Report a margin (best vs runner-up). The LLM is optional and only used for ambiguous diffs, constrained to the legal-move list |
| 3 | Jev (TypeSafe) cloud pre-check for "hand over board / changed?" | Network round trip on the hot path, another API key | Local stability gate (motion energy). Hands cause motion; a settled scene is the trigger |
| 4 | LLM locates board every 5 s for the overlay | Cost; overlay lags between answers | Local corner tracker: global translation by block matching on a downscaled frame, then per-corner template refinement, validated by a checkerboard score. Bootstrap from a manual 4-click calibration in the UI (free) or one LLM call |
| 5 | Stockfish launched + handshaken per best-move request | 10–100 ms process + handshake overhead each time; cold hash | One persistent engine actor. The timeout-then-never-reuse rule is kept: a timed-out process is killed and replaced, and a request is retried once |
| 6 | Fixed 0.5 s think every time | Wasted while the opponent thinks | **Pondering.** After suggesting move M with expected reply P, the engine searches the position after M,P during the opponent's turn. If the opponent plays P, `ponderhit` answers immediately. Otherwise `stop` and do a normal search with a warm hash |
| 7 | `say` subprocess per utterance into BlackHole | Process spawn + synthesis latency, macOS-only | Closed vocabulary (pieces, 64 squares, ~10 words) pre-rendered once to WAV via the platform TTS, then concatenated in memory and played to a named output device (BlackHole on macOS, VB-Cable on Windows). Unknown phrases fall back to whole-phrase synthesis, then get cached |
| 8 | Overlay drawn server-side, encoded with ffmpeg, MJPEG | CPU per frame, ffmpeg dependency | Overlay drawn **client-side** on a `<canvas>` from the snapshot (corners + suggestion) over `/frame.jpg`. JPEG encoding happens only when someone is watching, once per frame, and is shared by all viewers. Recording to file is deferred (see below) |
| 9 | Game loop = sequential poll loop with sleeps | Latency floor = sleep interval; controls wait for the cycle | Event-driven actor: `tokio::select!` over new frame / control / engine reply / LLM reply. Controls apply immediately, between frames |
| 10 | Event bus: 300 in memory, JSONL file, listeners | Fine | Kept as is. `tokio::broadcast` plus a ring buffer; subscribe is atomic with the history copy so there are no gaps or duplicates |
| 11 | macOS-only (WhatsApp Mac window, `say`, Keychain) | — | Platform seams as traits: `FrameSource` (window capture via `xcap` on macOS/Windows/Linux, video file via ffmpeg pipe, image dir), `Speaker`, keys via `keyring` (macOS Keychain / Windows Credential Manager) |
| 12 | Spending cap pauses the game | Fine | Kept. LLM use is now rare, so the cap is rarely hit |
| 13 | `chess.js` | — | `shakmaty` (fast legal movegen, SAN, FEN) |
| 14 | Server on 127.0.0.1:8000–8010, JSON + same-origin controls, 15 s keepalive | Fine | Kept as is (axum) |

## Measured (Windows 11, release build)

| What | Result | Source |
|---|---|---|
| Vision per frame (track + warp + gate + stats), 1280×720 | p50 3.3 ms, p95 3.4 ms | `chessglass bench` |
| Move reading, 40-ply synthetic game with head drift + noise | 39 correct, 0 wrong, 1 deferred (an under-promotion, by design) | `chessglass bench` |
| Stockfish 19, normal search / ponder hit | 300 ms / **0.9 ms** (10 of 10 hits) | `chessglass bench` |
| End to end through `run` (capture → loop → engine → speech), self-play game | **8.2 ms** from settled board to speech on a ponder hit | `bench --dump --selfplay` + `run --source dir:` |
| Speech: `say()` → audio queued | ~0.12 ms (78-word vocabulary pre-rendered in 0.7 s) | cg-voice example |
| Window capture (WGC) | ~13 ms/frame steady state | cg-capture `grab` example |

## Changes made during the build

- **Promotions.** A board photo shows that a pawn promoted but not which piece it became. Vision reports this as "ambiguous". The game loop then asks the LLM, which picks among the four pieces if one is configured. Without an LLM it assumes a queen and logs an "undo if it was another piece" note.
- **Gate: local-change criterion.** A piece appearing between two frames barely moves the *mean* motion. The gate now also treats any single thumbnail cell changing by more than 20 gray levels as motion. This took the bench from 24 to 39 of 40 moves read.
- **Tracker.** Each corner is predicted from its own recent motion, and the templates are sampled rotated by the board's roll since calibration. Templates are still never re-cut, so there is no drift. Error stays at or below 1.7 px through a 10° roll.
- **Capture on Windows** uses Windows Graphics Capture (xcap `wgc`). It is about 3× faster than GDI and captures GPU-composited call video, which GDI often returns black for.
- **Stockfish 19** has no `Ponder` UCI option. `go ponder` and `ponderhit` work without it.

## Process layout

```
capture thread ──watch<Frame>──┐
                               ▼
                 ┌──────── game actor (tokio task) ────────┐
controls ─mpsc──▶│ GameState (shakmaty), Vision (gate,      │──▶ Bus (events, snapshots) ──▶ SSE /events
                 │ tracker, reference board), budget        │
                 └──┬──────────────▲───────────┬────────────┘
                    │ requests      │ replies   │ say()
                    ▼               │           ▼
              Engine actor (UCI, ponder)    Speaker thread (clips → audio device)
              LLM client (optional, async)
```

There is one owner of game state, so the hot path takes no locks. All heavy per-frame work (~1–3 ms) runs inline in the actor.

## Crates

| crate | role | spec |
|---|---|---|
| `cg-core` | shared types: `Frame`, `FrameSource`, `GameEvent`, `Snapshot`, `Control`, `Bus`, `spoken_san` | (source is the spec) |
| `cg-vision` | homography, warp, square features, move inference, stability gate, corner tracker, synthetic renderer for tests | `docs/specs/cg-vision.md` |
| `cg-engine` | find / download / drive Stockfish; ponder; crash/timeout recovery | `docs/specs/cg-engine.md` |
| `cg-capture` | `FrameSource` impls: window, video file, image dir | `docs/specs/cg-capture.md` |
| `cg-voice` | `Speaker` impls: clip cache, command fallback, null | `docs/specs/cg-voice.md` |
| `cg-llm` | optional vision LLM (Anthropic / OpenAI): locate board, pick move among legal moves; cost tally | `docs/specs/cg-llm.md` |
| `cg-server` | axum command center, SSE, frame JPEG, overlay page, controls | `docs/specs/cg-server.md` |
| `cg-app` | `chessglass` binary: CLI, orchestrator/game actor, doctor, setup, bench | `docs/specs/cg-app.md` |

## Deferred / dropped

- **Persistent WGC session** (frame-arrived callbacks, ~1 ms/frame instead of a capture session per frame) and a macOS ScreenCaptureKit stream. These belong behind the existing `FrameSource` trait.

- **Overlay recording/streaming to file** (`--overlay-out`): dropped from v1. The overlay is an HTML page, so OBS Browser Source still works and covers streaming. It can come back later as an ffmpeg pipe fed the raw frames.
- **Jev / TypeSafe**: dropped (replaced by the local gate).
- **`tested.json` WhatsApp version check**: replaced by `doctor` capturing one frame from the window.
