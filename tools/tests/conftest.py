import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(TOOLS))  # so `import kibitz_tools` works without installing


def read_pgm(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


@pytest.fixture(scope="session")
def golden():
    """Golden output of crates/cg-vision/tests/golden.rs."""
    if not (GOLDEN / "golden.json").exists():
        pytest.skip("no golden data; run `cargo test -p cg-vision --test golden -- --ignored`")
    meta = json.loads((GOLDEN / "golden.json").read_text())
    meta["frame_a"] = read_pgm(GOLDEN / "frame_a.pgm")
    meta["frame_b"] = read_pgm(GOLDEN / "frame_b.pgm")
    for case in meta["cases"]:
        sq = case["sq_px"]
        case["warped_a"] = read_pgm(GOLDEN / f"warped_a_{sq}.pgm")
        case["warped_b"] = read_pgm(GOLDEN / f"warped_b_{sq}.pgm")
    return meta
