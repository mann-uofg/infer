"""Gets the WikiText-2 (raw) test set and tokenizes it for tools/perplexity.

    python scripts/prepare_wikitext.py --text-out models/wiki.test.raw
    python scripts/prepare_wikitext.py --text-out models/wiki.test.raw \\
        --tokenizer models/llama-3.2-1b --tokens-out models/llama-3.2-1b/wikitext2-test.bin

The text is the wiki.test.raw file from the same archive llama.cpp's scripts download
(ggml-org/ci on Hugging Face), byte for byte, so llama-perplexity (-f) reads exactly what this
tokenizes. (The Hugging Face "wikitext" parquet files drop the blank lines, which changes the
token stream.) The token file is that text through the Hugging Face tokenizer with a BOS token
at the start, as raw little-endian int32.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, help="directory with tokenizer files")
    ap.add_argument("--tokens-out", type=Path)
    args = ap.parse_args()

    archive = hf_hub_download(
        repo_id="ggml-org/ci", repo_type="dataset", filename="wikitext-2-raw-v1.zip"
    )
    with zipfile.ZipFile(archive) as z:
        raw = z.read("wikitext-2-raw/wiki.test.raw")
    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.write_bytes(raw)
    text = raw.decode("utf-8")
    print(f"wrote {args.text_out}: {len(raw)} bytes, {text.count(chr(10))} lines")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ids = tok(text, add_special_tokens=True)["input_ids"]
        out = args.tokens_out or args.tokenizer / "wikitext2-test.bin"
        np.asarray(ids, dtype="<i4").tofile(out)
        print(f"wrote {out}: {len(ids)} tokens (first {ids[0]}, bos {tok.bos_token_id})")


if __name__ == "__main__":
    main()
