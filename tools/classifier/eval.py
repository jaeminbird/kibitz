"""Confusion matrix of the square classifier.

    python classifier/eval.py data/crops.npz --model classifier/model.onnx [--games val|all] [--png cm.png]

`--model` is a .onnx (run with onnxruntime, i.e. what the Rust side will run)
or a .pt checkpoint. With a .pt, `--games val` (default) uses the games the
checkpoint held out; with a .onnx, or `--games all`, every crop is used.
Board-level accuracy (all 64 squares of a frame right) is printed too, since
that is what reading an arbitrary position needs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kibitz_tools.crops import CROP_H, CROP_W, normalise  # noqa: E402
from kibitz_tools.labels import CLASSES  # noqa: E402


def predict(model_path: Path, x: np.ndarray) -> np.ndarray:
    if model_path.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return np.concatenate([sess.run(["logits"], {"crops": x[i : i + 512]})[0] for i in range(0, len(x), 512)])
    import torch

    from model import load

    m, _ = load(model_path)
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(x[i : i + 512])).numpy() for i in range(0, len(x), 512)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", type=Path, default=Path(__file__).with_name("model.onnx"))
    ap.add_argument("--checkpoint", type=Path, help="a .pt whose val_games select the crops (with an .onnx --model)")
    ap.add_argument("--games", choices=["val", "all"], default="val")
    ap.add_argument("--png", type=Path, help="also save the matrix as an image")
    args = ap.parse_args()

    d = np.load(args.dataset)
    keep = np.ones(len(d["labels"]), bool)
    ck_path = args.checkpoint or (args.model if args.model.suffix == ".pt" else None)
    if args.games == "val" and ck_path is not None:
        import torch

        val = torch.load(ck_path, map_location="cpu", weights_only=False)["val_games"]
        keep = np.isin(d["game"], val)
        print(f"val games: {val}")
    elif args.games == "val":
        print("no checkpoint to name the val games: using all crops (pass --checkpoint)")
    idx = np.nonzero(keep)[0]

    j = int(d["jitter"])
    c = d["crops"][idx][:, j : j + CROP_H, j : j + CROP_W]
    x = normalise(c, d["board_mean"][idx], d["board_std"][idx])[:, None]
    y = d["labels"][idx].astype(int)
    p = predict(args.model, x).argmax(1)

    k = len(CLASSES)
    conf = np.zeros((k, k), np.int64)
    np.add.at(conf, (y, p), 1)
    print(f"\n{len(y)} crops, accuracy {np.trace(conf) / conf.sum():.4f}   (rows = truth, cols = predicted)")
    print("       " + "".join(f"{c:>6}" for c in CLASSES) + "  recall")
    for i, c in enumerate(CLASSES):
        rec = conf[i, i] / conf[i].sum() if conf[i].sum() else float("nan")
        print(f"{c:>6} " + "".join(f"{v:>6}" for v in conf[i]) + f"  {rec:.3f}")

    # Frames where every square is right.
    frames = d["frame_ts"][idx].astype(np.int64)
    games = d["game"][idx]
    ok = {}
    for f, g, good in zip(frames, games, y == p):
        ok[(g, f)] = ok.get((g, f), True) and bool(good)
    print(f"\nboards fully correct: {sum(ok.values())}/{len(ok)}")

    if args.png:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(np.log1p(conf), cmap="Blues")
        ax.set_xticks(range(k), CLASSES)
        ax.set_yticks(range(k), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("truth")
        for i in range(k):
            for jj in range(k):
                if conf[i, jj]:
                    ax.text(jj, i, conf[i, jj], ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
