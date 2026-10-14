"""Runs the C++ engine binary and reads back what it dumped."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "build" / "release" / "infer"


def run_engine(model_dir: Path, tokens: list[int], *extra: str) -> subprocess.CompletedProcess:
    cmd = [str(BINARY), "--model", str(model_dir), "--tokens", ",".join(map(str, tokens)), *extra]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def engine_dump(model_dir: Path, tokens: list[int], num_layers: int, hidden: int, vocab: int,
                *extra: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Hidden states (num_layers + 1 arrays of [T, H]) and logits [T, V] from --dump."""
    t = len(tokens)
    with tempfile.TemporaryDirectory() as d:
        run_engine(model_dir, tokens, "--dump", d, *extra)
        states = [
            np.fromfile(Path(d) / f"hidden_{i:02d}.f32", dtype=np.float32).reshape(t, hidden)
            for i in range(num_layers + 1)
        ]
        logits = np.fromfile(Path(d) / "logits.f32", dtype=np.float32).reshape(t, vocab)
    return states, logits


def engine_generate(model_dir: Path, tokens: list[int], n: int, *extra: str) -> list[int]:
    out = run_engine(model_dir, tokens, "--generate", str(n), *extra)
    return [int(x) for x in out.stdout.split()]
