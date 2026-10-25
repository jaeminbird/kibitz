"""Export a train.py checkpoint to ONNX and check onnxruntime agrees with torch.

    python classifier/export_onnx.py classifier/model.pt -o classifier/model.onnx [--dataset data/crops.npz]

The check runs both models on real crops from `--dataset` if given (centre
window, normalised as in MODEL_CONTRACT.md), else on random input, and fails
if any logit differs by more than 1e-3 or any argmax differs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kibitz_tools.crops import CROP_H, CROP_W, normalise  # noqa: E402

from model import load  # noqa: E402

# The dynamo exporter's minimum. ONNX Runtime 1.28 (ort 2.0.0-rc.13) supports it.
OPSET = 18


def sample_inputs(dataset: Path | None, n: int = 256) -> np.ndarray:
    if dataset is None:
        return np.random.default_rng(0).normal(size=(n, 1, CROP_H, CROP_W)).astype(np.float32)
    d = np.load(dataset)
    j = int(d["jitter"])
    idx = np.random.default_rng(0).choice(len(d["labels"]), min(n, len(d["labels"])), replace=False)
    c = d["crops"][idx][:, j : j + CROP_H, j : j + CROP_W]
    return normalise(c, d["board_mean"][idx], d["board_std"][idx])[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_name("model.onnx"))
    ap.add_argument("--dataset", type=Path)
    args = ap.parse_args()
    # The torch exporter prints emoji, which a cp1252 Windows console can't encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model, ck = load(args.checkpoint)
    dummy = torch.zeros(1, 1, CROP_H, CROP_W)
    # Batch dimension is dynamic: the runtime sends all 64 squares at once.
    torch.onnx.export(
        model, (dummy,), args.out, input_names=["crops"], output_names=["logits"],
        opset_version=OPSET, dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        dynamo=True, external_data=False,  # one self-contained file for the Rust side
    )
    import onnx

    opset = onnx.load(args.out).opset_import[0].version
    print(f"wrote {args.out} (opset {opset})")

    x = sample_inputs(args.dataset)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"crops": x})[0]
    err = np.abs(ref - got).max()
    same = np.mean(ref.argmax(1) == got.argmax(1))
    print(f"onnxruntime vs torch on {len(x)} crops: max |logit diff| {err:.2e}, argmax agreement {same:.4f}")
    # Also a batch of exactly 64, the runtime's shape.
    got64 = sess.run(["logits"], {"crops": x[:64]})[0]
    assert got64.shape == (64, len(ck["classes"]))
    if err > 1e-3 or same < 1.0:
        sys.exit("MISMATCH between torch and onnxruntime")
    print("OK")


if __name__ == "__main__":
    main()
