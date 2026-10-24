# cg-app spec (`kibitz` binary)

## CLI (clap derive)
```
kibitz run     [--color white|black] [--fen FEN] [--source window:WhatsApp|file:PATH|dir:PATH]
                   [--fps 15] [--movetime-ms 300] [--settle-ms 300] [--no-ponder]
                   [--model claude-opus-5-5] [--no-llm] [--max-cost 10]
                   [--voice-device BlackHole] [--no-voice] [--run-dir DIR] [--log FILE] [--no-open]
kibitz setup   [--key-anthropic K] [--key-openai K]   # download Stockfish, warm voice clips, store keys
kibitz doctor                                          # check every dependency, print fixes
kibitz bench   [--frames 300]                          # synthetic end-to-end latency benchmark
```
Env overrides: `KIBITZ_STOCKFISH`, `KIBITZ_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## Modules
- `game.rs`: `GameState`, a pure wrapper over shakmaty: start position, move list (SAN + UCI + side), play/undo, outcome, PGN text, engine `Position`. Unit-tested.
- `orchestrator.rs`: the game actor (see ARCHITECTURE.md, "Process layout").
- `doctor.rs`, `setup.rs`, `bench.rs`, `main.rs`.

## Game actor
State: `GameState`, `paused`, `corners: Option<Corners>`, `CornerTracker`, `StabilityGate`, `reference: Option<SquareStats>` (taken at the last commit), `suggestion`, `spent`, `pending_engine: Option<(pos_id, JoinHandle)>`, `llm_busy`, `status`, latency marks.

Per new frame (watch changed):
1. If paused / game over → skip.
2. If no corners → `LookingForBoard`. If the LLM is enabled and not busy and ≥ 5 s since the last attempt → spawn `locate_board`. Manual `SetCorners` is always accepted.
3. Track: `tracker.track(frame)`. None for > 1.5 s → `LostTrack("board not visible")`, clear corners (falls back to step 2).
4. Warp the board (sq_px 32) and push the **warped board** into the gate. The gate looks only at the board, so movement elsewhere in the room doesn't block reading.
5. If there is no reference yet and the gate is Stable → reference = stats(warped); status = turn status; if it's my turn, request a suggestion.
6. On `Stable{fresh:true}`: diff vs reference → `infer_move`. Emit a `Frame` event.
   - accept → commit (side = whoever is to move), reference = current stats, then go to "after commit".
   - "no change" → nothing.
   - "ambiguous" with LLM → `pick_move` among the top ranked candidates (SAN). Answer in the list → commit with source "llm".
   - "unexplained change" persisting across 3 consecutive fresh-stable reads → `LostTrack("board doesn't match")`.
7. After commit: game over → announce and stop. If it's now my turn → request a suggestion (the engine handles ponder hits). If it's now the opponent's turn and the last suggestion had a ponder move → `engine.ponder(position + ponder move)`.

Suggestion reply (tagged with a position id; discard if stale): build `Suggestion` (SAN via shakmaty, `spoken_san`, score), `speaker.say(spoken)`, emit `Suggest`, update the latency numbers.

Controls: Pause/Resume, Undo (pop the move, engine.stop, reference = None so it re-baselines on the next stable frame), Move (commit manual, reference = None), NewGame, Repeat, SetCorners (`Corners::from_view` × frame size, `tracker.reset`, `gate.reset`, reference = None), Relocate (corners = None).

Budget: after every LLM call add the cost. When `spent ≥ max_cost` → pause with "Spending limit reached. Paused." (spoken once).

Run folder: `game.pgn` rewritten atomically after each commit; `frames/<ts>-<what>.jpg` on commit / lost track.

Publish a snapshot after every state change (and at most every 250 ms for frame/latency counters).
