"""Offline tooling for kibitz: load recorded runs, re-run the board vision in
numpy, and build data for tuning and for the piece classifier.

Nothing here is on the runtime path. `board.py` mirrors the Rust code in
`crates/cg-vision/src/{geometry,features}.rs` and is checked against it by
`tests/test_parity.py`.
"""
