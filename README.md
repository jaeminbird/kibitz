# kibitz

Reads an over-the-board chess game from a video call (e.g. Meta glasses on WhatsApp), picks the best move with Stockfish, and speaks it into the call. It is a Rust rewrite of *chesshacker*, built for latency: about 8 ms from the board settling to the move being spoken when the engine's ponder prediction hits.

Like the original, it is for friendly games where everyone at the board knows about it, **not for rated or tournament play.**

```sh
cargo build --release
target/release/kibitz setup        # Stockfish 19 (checksum-pinned) + voice clips
target/release/kibitz doctor       # what's missing and how to fix it
target/release/kibitz run --color white --voice-device "CABLE Input"   # or "BlackHole"
target/release/kibitz bench        # latency report on synthetic frames
```

In the command center (opens automatically), click **Set corners** and click the board's four corners. With an `ANTHROPIC_API_KEY`, the board is found automatically instead, and an LLM is used only to break ties.

See `docs/ARCHITECTURE.md` for the design and how it compares with the original, and `docs/specs/` for per-crate contracts.
