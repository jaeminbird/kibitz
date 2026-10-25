# tools/: offline Python for kibitz

Tooling that sits next to the Rust workspace. **Nothing here is on the runtime path.** It reads
what `kibitz run --run-dir DIR` records, re-runs the board vision in numpy to tune thresholds
(`docs/IMPROVEMENTS.md` section 2), and trains the per-square piece classifier (section 4.1).

```
kibitz_tools/      shared package
  rundir.py        load a run dir: frames + sidecars, game.pgn, events.jsonl
  board.py         numpy port of warp_board, square_rect, square mapping, square_stats/diff
  labels.py        FEN -> 64 labels (13 classes), changed squares between positions
  analysis.py      per-square diffs of consecutive commit frames + ground truth
  crops.py         classifier crop geometry (the model contract)
  fake_run.py      run dirs from `kibitz bench --dump` (synthetic, for smoke tests)
lab/
  vision_lab.ipynb frames + corners, warped boards, diff heatmaps, diff distributions
  sweep.py         min_change threshold sweep (TPR/FPR) over run dirs
classifier/
  build_dataset.py run dirs -> square crops .npz (labels from commit frames)
  model.py         ~290K-param CNN
  train.py         training, split by game, with augmentation
  export_onnx.py   -> model.onnx, checked against torch with onnxruntime
  eval.py          confusion matrix
  MODEL_CONTRACT.md  exact input/output for the Rust `ort` side
tests/             pytest: warp/stats parity with Rust, mapping, labels, run dirs, crops
  golden/          written by crates/cg-vision/tests/golden.rs
```

## Setup (Windows, Python 3.14, no uv)

From `tools/`:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
# classifier only (CPU torch, ~200 MB):
.venv\Scripts\python -m pip install -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

PyTorch has CPU wheels for Python 3.14 on Windows (verified with torch 2.14.1+cpu). If a future
combination has none, create a separate 3.12 venv for the classifier only
(`py -3.12 -m venv .venv-ml`, then install `requirements-ml.txt` into it). The rest works on 3.14.

On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python`.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

`test_parity.py` compares the numpy ports with golden output from the Rust code. Warp: ±1 grey
level allowed, bit-exact in practice. Stats and diff: within 1e-3, about 6e-8 in practice. Regenerate
the golden files after changing `geometry.rs` or `features.rs`:

```powershell
cargo test -p cg-vision --test golden -- --ignored
```

## Data

Real footage: `kibitz run --run-dir data\game1 --log data\game1\events.jsonl ...`.
**Only `"commit"` frames are labels.** `"lost"` and `"missed"` frames are kept for inspection, but
their `fen` may not match the board.

Synthetic stand-in (until there is real footage). `bench --dump` writes PNGs without labels;
`fake_run` reconstructs the labels and corners and writes proper run dirs:

```powershell
cargo build --release
.venv\Scripts\python -m kibitz_tools.fake_run --kibitz ..\target\release\kibitz.exe --plies 60 --runs 2 --out data\synth
```

Without Stockfish only the scripted game is available (`--selfplay` needs it), so `--runs 2`
splits that one game into two run dirs to exercise the split by game. The diffs come from JPEG q80
frames (as saved by `--run-dir`), so they are noisier than what the live pipeline sees.

## Vision lab

```powershell
.venv\Scripts\python lab\sweep.py data\synth\run0 data\synth\run1
.venv\Scripts\jupyter lab lab\vision_lab.ipynb     # edit RUNS in the first cell
```

The sweep applies the runtime threshold rule `t = max(min_change, 2.5 × median diff)`
(`excess_over_threshold`, infer.rs), or `min_change` alone with `--no-median`. It prints the Rust
default next to the candidates. That default is hard-coded in `sweep.py` and the notebook, so update
them if `InferConfig::default` changes.

## Classifier

```powershell
.venv\Scripts\python classifier\build_dataset.py data\synth\run0 data\synth\run1 -o data\crops.npz
.venv\Scripts\python classifier\train.py data\crops.npz -o classifier\model.pt
.venv\Scripts\python classifier\export_onnx.py classifier\model.pt -o classifier\model.onnx --dataset data\crops.npz
.venv\Scripts\python classifier\eval.py data\crops.npz --model classifier\model.onnx --checkpoint classifier\model.pt
```

`train.py` needs at least two games (run dirs) and holds out about 20% of them (`--val-games`).
`data/`, `*.pt` and `*.onnx` are gitignored.
