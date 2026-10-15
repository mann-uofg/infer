"""tools/perplexity against the reference implementation of the same method.

Random tokens on the tiny random model: the perplexity is meaningless as a number, but both
sides must compute the same one, for every weight format.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reference.llama import Llama
from reference.perplexity import chunked_nll
from tests.python.engine_harness import REPO

BINARY = REPO / "build" / "release" / "perplexity"
TINY = REPO / "models" / "tiny"

pytestmark = pytest.mark.skipif(
    not BINARY.exists() or not torch.cuda.is_available() or not (TINY / "config.json").exists(),
    reason="needs the built tool, a GPU and models/tiny",
)


@pytest.mark.parametrize("weights,group", [("bf16", 128), ("int8", 128), ("int4", 128),
                                           ("int4", 32)])
def test_matches_reference(weights, group):
    ctx, chunks, bos = 64, 3, 1
    tokens = np.random.default_rng(5).integers(0, 4096, ctx * chunks + 17).astype("<i4")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tokens.bin"
        tokens.tofile(path)
        out = subprocess.run(
            [str(BINARY), "--model", str(TINY), "--tokens", str(path), "--ctx", str(ctx),
             "--bos", str(bos), "--weights", weights, "--int4-group", str(group)],
            check=True, capture_output=True, text=True,
        ).stdout
    got = float(re.search(r"mean nll ([0-9.]+)", out).group(1))
    assert int(re.search(r"tokens scored (\d+)", out).group(1)) == chunks * (ctx - 1 - ctx // 2)

    ref = Llama(TINY, device="cuda", kv_dtype=torch.bfloat16, weights=weights,
                int4_group=group)
    total, count = chunked_nll(lambda c: ref.forward(c), tokens.tolist(), ctx, bos=bos)
    assert got == pytest.approx(total / count, rel=1e-4)
