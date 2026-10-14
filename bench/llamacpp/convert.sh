#!/usr/bin/env bash
# Converts a Hugging Face checkpoint to GGUF and makes the quantized variants used in the
# comparison: BF16, Q8_0 (closest to my int8) and Q4_0 (closest to my int4 g128).
#
#   bench/llamacpp/convert.sh MODEL_DIR OUT_DIR [LLAMA_CPP_DIR]
#
# Real checkpoints go through llama.cpp's converter unchanged. The random-weight checkpoints
# have no tokenizer, so they get GPT-2's, and the converter is told to use its plain BPE path
# for it (it would otherwise look for a Llama 3 tokenizer). The converter pads the vocabulary
# to the model's 128256 rows, so every tensor has exactly the real model's shape; the text it
# would produce is meaningless, which does not matter for speed.
set -euo pipefail

model=$(realpath "${1:?model dir}")
out=${2:?out dir}
llama=$(realpath "${3:-$HOME/src/llama.cpp}")
py="${PYTHON:-$HOME/venvs/infer/bin/python}"
mkdir -p "$out"

if [ -f "$model/tokenizer.json" ]; then
  "$py" "$llama/convert_hf_to_gguf.py" "$model" --outtype bf16 --outfile "$out/model-bf16.gguf"
else
  src=$(mktemp -d)
  trap 'rm -rf "$src"' EXIT
  ln -s "$model"/* "$src"/
  "$py" - "$src" "$llama" "$out/model-bf16.gguf" <<'EOF'
import runpy
import shutil
import sys

from huggingface_hub import hf_hub_download

src, llama, outfile = sys.argv[1:4]
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    shutil.copy(hf_hub_download("openai-community/gpt2", f), src)

sys.path.insert(0, llama)
import conversion.llama  # noqa: E402

conversion.llama.LlamaModel.set_vocab = lambda self: self._set_vocab_gpt2()
sys.argv = ["convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", outfile]
runpy.run_path(f"{llama}/convert_hf_to_gguf.py", run_name="__main__")
EOF
fi

for q in Q8_0 Q4_0; do
  "$llama/build/bin/llama-quantize" "$out/model-bf16.gguf" "$out/model-$q.gguf" "$q" > /dev/null 2>&1
done
ls -la "$out"
