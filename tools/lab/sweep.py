"""Threshold sweep for the per-square change score (IMPROVEMENTS.md section 2).

For every pair of consecutive commit frames in the given run dirs, compute the
per-square diff the runtime would see and compare it with the squares that
really changed (python-chess). For each candidate `min_change`, report how many
truly changed squares exceed the threshold (want all) and how many unchanged
squares do (want none), using the runtime's threshold rule
t = max(min_change, 2.5 * median diff) unless --no-median.

    python lab/sweep.py data/run1 data/run2 [--sq-px 32] [--no-median]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kibitz_tools.analysis import pair_diffs, threshold  # noqa: E402
from kibitz_tools.rundir import load_run  # noqa: E402

# Current Rust default: `InferConfig::default().min_change` in
# crates/cg-vision/src/infer.rs. Update this if that changes.
RUST_MIN_CHANGE = 0.2

CANDIDATES = [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sq-px", type=int, default=32)
    ap.add_argument("--no-median", action="store_true", help="threshold = min_change only")
    args = ap.parse_args()

    pairs = []
    for r in args.runs:
        ps = pair_diffs(load_run(r), args.sq_px)
        print(f"{r}: {len(ps)} commit pairs")
        pairs += ps
    if not pairs:
        sys.exit("no consecutive commit pairs found")

    changed = np.concatenate([p.diff[p.changed] for p in pairs])
    unchanged = np.concatenate([p.diff[~p.changed] for p in pairs])
    print(f"\n{len(pairs)} pairs, {changed.size} changed squares, {unchanged.size} unchanged squares")
    q = [0, 1, 5, 50, 95, 99, 100]
    print("changed   diff percentiles " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(q, np.percentile(changed, q))))
    print("unchanged diff percentiles " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(q, np.percentile(unchanged, q))))
    gap = changed.min() - unchanged.max()
    print(f"gap (min changed - max unchanged): {gap:+.3f}" + ("  (overlap!)" if gap <= 0 else ""))

    factor = 0.0 if args.no_median else 2.5
    rule = "t = min_change" if args.no_median else "t = max(min_change, 2.5 * median)"
    print(f"\nthreshold rule: {rule}; Rust default min_change = {RUST_MIN_CHANGE} (InferConfig::default, infer.rs)")
    print(f"{'min_change':>10} {'changed>t':>14} {'TPR':>7} {'unchanged>t':>14} {'FPR':>8} {'clean pairs':>12}")
    for mc in sorted(set(CANDIDATES + [RUST_MIN_CHANGE])):
        tp = fp = clean = 0
        for p in pairs:
            above = p.diff > threshold(p.diff, mc, factor)
            tp += int((above & p.changed).sum())
            fp += int((above & ~p.changed).sum())
            # A pair is clean when exactly the true squares are above threshold
            # (what 2.5's "coverage" rule and a zero residual both need).
            clean += int(np.array_equal(above, p.changed))
        mark = "  <- Rust default" if mc == RUST_MIN_CHANGE else ""
        print(
            f"{mc:>10.3f} {tp:>7}/{changed.size:<6} {tp / changed.size:>7.3f} "
            f"{fp:>7}/{unchanged.size:<6} {fp / unchanged.size:>8.4f} {clean:>5}/{len(pairs):<6}{mark}"
        )


if __name__ == "__main__":
    main()
