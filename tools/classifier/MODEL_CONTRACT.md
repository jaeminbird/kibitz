# Square classifier: model contract

What the Rust side (`ort`, IMPROVEMENTS.md 4.1) must do to run `model.onnx`.
The reference implementation is `tools/kibitz_tools/crops.py`. `tests/test_crops.py`
pins the geometry. If you change anything here, change that file and retrain.

## Files

- `model.onnx`: a single self-contained file (no external `.data`), opset 18.
  Produced by `classifier/export_onnx.py`, which checks onnxruntime against torch
  (max logit difference must be ≤ 1e-3, and every argmax must match).

## Input

| | |
|---|---|
| name | `crops` |
| shape | `[batch, 1, 56, 48]` (N, C, H, W); `batch` is dynamic (1 to 1024). Send all 64 squares as one batch of 64 |
| dtype | `float32` |
| channel | grayscale, `Frame::gray` (BT.601 `(77 r + 150 g + 29 b) >> 8`) |

### Crop geometry

1. **Warp.** Use the same homography as `warp_board` with `sq_px = 32`: board
   pixel coordinates `(u, v)` in `[0, 256)²`, corners `[a8, h8, h1, a1]` → (0,0), (256,0),
   (256,256), (0,256). Sampling is identical to `geometry.rs`: the pixel centre `(u + 0.5, v + 0.5)`
   is mapped to the image, minus 0.5, sampled bilinearly, with out-of-frame pixels = 0, and the result
   rounded half away from zero to `u8`. The crops reach **outside** the 8×8 board (up to 16 px
   above, 8 px to the sides and below), so the warp must be evaluated for `u, v` in `[-16, 272)`.
   This is the same formula with `u, v` allowed to be negative or ≥ 256. Rust's `warp_board`
   only covers `[0, 256)` today, so add an offset or padding parameter.
2. **Orient to the player.** The warped board is White's view (row 0 = rank 8). If the player
   is **Black**, rotate it 180°, i.e. use `(u', v') = (255 - u, 255 - v)`. After this the player's
   own side is always at the bottom of the image.
3. **Crop.** Let the square's top-left in the oriented 256×256 board be `(x, y)`:
   - White: `row = 7 - rank`, `col = file`, so `x = 32·col`, `y = 32·row`.
   - Black: `x = 32·(7 - col)`, `y = 32·(7 - row)` (`row`, `col` as for White).

   The crop is the oriented board pixels `[x - 8, x + 40) × [y - 16, y + 40)` (x range × y range):
   8 px of context left, right and below, and 16 px above. Tall pieces lean upward in the image,
   toward the far side. Result: 56 rows × 48 columns.
4. **Normalise.** Compute `mean` and `std` over the **256×256 board only**, oriented or not,
   without the context border: population std, floored at 1.0. This is the same as the global
   moments in `square_stats` (`features.rs`). Then `x = (pixel - mean) / std` as float32. Use the
   same `mean` and `std` for all 64 crops of a frame.

### Square order

Batch index `i` = square `i` in shakmaty numbering: a1 = 0, b1 = 1, … h1 = 7, a2 = 8, … h8 = 63.
The order does not change with the player's colour; only the pixels are rotated.

## Output

| | |
|---|---|
| name | `logits` |
| shape | `[batch, 13]` |
| dtype | `float32`, raw logits (apply softmax for probabilities) |

Class index (`tools/kibitz_tools/labels.py`, `CLASSES`):

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| class | empty | P | N | B | R | Q | K | p | n | b | r | q | k |

Upper case = white, lower case = black. Within each colour the order is shakmaty's `Role`
order (pawn, knight, bishop, rook, queen, king), so for a non-empty class
`color = if idx <= 6 { White } else { Black }` and `role = Role::ALL[(idx - 1) % 6]`.

## Training data

Only "commit" frames from `--run-dir` are labels; "lost" and "missed" frames are never used.
The split is by game (run dir). The current `model.onnx` (if any) was trained on **synthetic
renders only** (discs with role rings, `cg-vision/src/synth.rs`). It is a pipeline smoke test,
not a usable model for real pieces.
