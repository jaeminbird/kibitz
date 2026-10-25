"""Train the square classifier on a dataset from build_dataset.py.

The split is by game (run dir), never by crop: crops from one game share the
board, lighting and pieces, so a per-crop split would leak.

    python classifier/train.py data/crops.npz -o classifier/model.pt [--val-games 0.2] [--epochs 15]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kibitz_tools.crops import CROP_H, CROP_W, normalise  # noqa: E402
from kibitz_tools.labels import CLASSES  # noqa: E402

from model import SquareNet  # noqa: E402


def split_games(games: np.ndarray, val_frac: float, seed: int) -> tuple[set, set]:
    """Shuffle the distinct games and hold out about `val_frac` of them (at least one)."""
    uniq = sorted(set(games.tolist()))
    if len(uniq) < 2:
        sys.exit("need at least 2 games (run dirs) to split by game")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * val_frac))
    return set(uniq[n_val:]), set(uniq[:n_val])


class Crops(Dataset):
    def __init__(self, d: dict, idx: np.ndarray, augment: bool):
        self.crops = d["crops"][idx]
        self.labels = d["labels"][idx].astype(np.int64)
        self.mean = d["board_mean"][idx]
        self.std = d["board_std"][idx]
        self.jitter = int(d["jitter"])
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        c, j, rng = self.crops[i], self.jitter, self.rng
        if self.augment:
            # Small shift: a random window from the jitter margin (corner error).
            y, x = rng.integers(0, 2 * j + 1, 2)
            c = c[y : y + CROP_H, x : x + CROP_W]
            if rng.random() < 0.5:
                c = c[:, ::-1]  # mirror: a piece's class doesn't depend on its facing
            # Local brightness/contrast. Normalisation below uses the board's
            # stored moments, so this simulates a shadow or glare on one square.
            f = (c.astype(np.float32) - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-30, 30)
            img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
            if rng.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
            if rng.random() < 0.3:  # call video is heavily compressed
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=int(rng.integers(20, 80)))
                img = Image.open(buf)
            c = np.asarray(img)
        else:
            c = c[j : j + CROP_H, j : j + CROP_W]
        x = normalise(c[None], self.mean[i], self.std[i])  # (1, H, W)
        return torch.from_numpy(np.ascontiguousarray(x)), self.labels[i]


def evaluate(model, loader) -> tuple[float, np.ndarray]:
    model.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    with torch.no_grad():
        for x, y in loader:
            p = model(x).argmax(1).numpy()
            np.add.at(conf, (y.numpy(), p), 1)
    return np.trace(conf) / max(conf.sum(), 1), conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_name("model.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-games", type=float, default=0.2, help="fraction of games held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = dict(np.load(args.dataset))
    train_g, val_g = split_games(d["game"], args.val_games, args.seed)
    tr = np.nonzero(np.isin(d["game"], list(train_g)))[0]
    va = np.nonzero(np.isin(d["game"], list(val_g)))[0]
    print(f"train: {len(train_g)} games, {tr.size} crops; val: {len(val_g)} games, {va.size} crops")
    train_dl = DataLoader(Crops(d, tr, True), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(Crops(d, va, False), batch_size=512)

    model = SquareNet(len(CLASSES))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    # About half the squares are empty: weight classes by 1/sqrt(frequency).
    freq = np.bincount(d["labels"][tr], minlength=len(CLASSES)).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(freq, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl))

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        acc, _ = evaluate(model, val_dl)
        print(f"epoch {ep + 1:>3}  loss {total / tr.size:.4f}  val acc {acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "input_hw": [CROP_H, CROP_W],
                    "train_games": sorted(train_g),
                    "val_games": sorted(val_g),
                    "val_acc": acc,
                },
                args.out,
            )
    print(f"best val acc {best:.4f}; saved {args.out}")


if __name__ == "__main__":
    main()
