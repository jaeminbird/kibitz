<div align="center">

# kibitz

**Like the kibitzer over your shoulder, kibitz sees the board through your Meta glasses and whispers the best move.**

Meta glasses watch the board. Stockfish thinks. The move is spoken in your ear.<br>
**~8 ms** from the board going still to the move being spoken.

[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-windows%20%7C%20macos%20%7C%20linux-lightgrey)
![Engine](https://img.shields.io/badge/engine-Stockfish%2019-success)

</div>

---

## What it does

You wear Meta glasses (or point any camera at a board) and share the video in a call to your computer. kibitz:

1. **sees** the board in the call window,
2. **reads** each move the moment the hand leaves the board,
3. **thinks** with Stockfish 19,
4. **speaks** the reply ("Knight to f3") into the call, so you hear it in your ear.

A local command center shows the live board, the suggestion, an eval bar, the move list and the latency of every step.

## Why it's fast

kibitz is a Rust rewrite of a Node/TypeScript prototype that sent every frame to a cloud vision model. The trick is to **never ask "what's on the board?"** kibitz already knows the position, so it only has to work out *which of ~30 legal moves* just happened. That can be done locally in microseconds.

| Stage | The usual way | kibitz | Measured |
|---|---|---|---|
| Read the move | cloud vision LLM, 1–10 s, $$ | per-square diff scored against legal moves only | **3.3 ms** / frame |
| "Is a hand in the way?" | cloud pre-check | local motion gate | **0.08 ms** |
| Follow the board as your head moves | re-locate with an LLM every 5 s | local corner tracker | **≤1.7 px** error at a 10° roll |
| Engine | spawn Stockfish per move, think 500 ms | one warm engine that **ponders on your opponent's time** | **0.9 ms** on a ponder hit |
| Speak | spawn `say` per phrase | pre-rendered word clips, mixed in memory | **0.12 ms** to queue |

Reading the move needs no API key and makes no paid calls. An LLM is optional and only bootstraps board detection or breaks ties.

<sub>Numbers come from `kibitz bench` and an end-to-end run on a release build (Windows 11). Over 40 plies of synthetic play with head drift and sensor noise: 39 read correctly, **0 wrong**, and 1 under-promotion deferred by design.</sub>

## Quick start

```sh
git clone https://github.com/jaeminbird/kibitz && cd kibitz
cargo build --release

./target/release/kibitz setup    # Stockfish 19 (SHA-256 pinned) + voice clips
./target/release/kibitz doctor   # tells you exactly what's missing
./target/release/kibitz run --color white --voice-device "CABLE Input"
```

Then, in the command center that opens: click **Set corners** and click the board's four corners. That's it.

<details>
<summary><b>Routing the voice into the call</b></summary>

kibitz speaks to an audio output device. Make that device the call's microphone:

- **Windows:** install [VB-CABLE](https://vb-audio.com/Cable/), run with `--voice-device "CABLE Input"`, and set the call's mic to *CABLE Output*.
- **macOS:** install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole), run with `--voice-device BlackHole`, and set the call's mic to *BlackHole 2ch*.
- Testing: leave out `--voice-device` and it plays on your speakers.

</details>

<details>
<summary><b>All commands</b></summary>

```
kibitz run     [--color white|black] [--fen FEN] [--source window:WhatsApp|file:clip.mp4|dir:frames/]
               [--fps 15] [--movetime-ms 300] [--settle-ms 300] [--no-ponder]
               [--model claude-opus-5-5] [--no-llm] [--max-cost 10]
               [--voice-device NAME] [--no-voice] [--run-dir DIR] [--log events.jsonl]
kibitz setup   [--key-anthropic KEY] [--key-openai KEY]
kibitz doctor  [--source window:WhatsApp]
kibitz bench   [--frames N] [--dump DIR [--selfplay]]
```

- `--source file:` replays a recorded game, which is great for tuning.
- `--run-dir` saves `game.pgn` and key frames, each with a JSON sidecar (board corners and FEN) for the offline tools in [`tools/`](tools/).
- `--max-cost` caps optional LLM spending; the game pauses itself at the cap.

</details>

## How it works

```
capture thread ──latest frame──▶ game actor ──▶ event bus ──▶ command center (SSE)
                                 │  track → warp → gate → diff → legal-move match
                                 ├──▶ Stockfish actor (persistent, go ponder / ponderhit)
                                 ├──▶ speaker thread (word clips → audio device)
                                 └──▶ vision LLM (optional: find board, break ties)
```

- **One owner of game state.** The hot path takes no locks, never sleeps, and never queues stale frames.
- **Safety rules carried over:** a timed-out engine is killed and never reused; controls are same-origin JSON only; the server listens on `127.0.0.1` only.
- **Platform seams are traits.** Capture is Windows Graphics Capture, macOS or X11/Wayland via `xcap`. Speech is SAPI, `say` or `espeak-ng`. Keys live in the OS keychain.
- **Offline tools are Python.** [`tools/`](tools/) has a vision lab (notebook plus threshold sweep over recorded `--run-dir` games) and a piece-classifier pipeline (dataset, PyTorch training, ONNX export). Its board warp and square stats are tested to match the Rust ones exactly. Nothing in Python runs during a game.

The deep dive, including every design decision compared with the original and what changed during the build, is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Per-crate contracts are in [`docs/specs/`](docs/specs).

## Status

| State | Item |
|---|---|
| Done | Full pipeline verified end to end on synthetic games; 134 Rust tests, 23 Python tests |
| Done | Windows build, capture, speech and engine |
| Testing | Real-camera tuning: thresholds come from synthetic data, so expect to tweak `--settle-ms` |
| Testing | macOS / Linux: written and cross-platform by design, not yet run there |
| Testing | Offline vision lab and piece-classifier pipeline: run on synthetic data, waiting on real recordings |
| Planned | Persistent capture session (~1 ms/frame), recorded-game regression suite, classifier in the live path via ONNX Runtime |

Issues and PRs welcome, especially recordings of real games to tune against: a `--run-dir` folder from a real game is exactly what [`tools/`](tools/) needs.

## How it was built

The architecture and contracts were planned by Claude Opus 5.5. The crates were implemented in parallel by Claude Haiku 5.5 agents against those specs, then integrated, benchmarked and debugged end to end.

## License

[MIT](LICENSE)
