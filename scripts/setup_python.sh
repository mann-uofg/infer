#!/usr/bin/env bash
# Creates the Python virtual environment and installs requirements.txt.
#
# The environment lives in the Linux home directory by default instead of the repo, because
# the repo sits on a Windows drive and a venv there is very slow under WSL.
# Needs the python3-venv package (sudo apt-get install python3.12-venv on Ubuntu 24.04).
set -euo pipefail

VENV="${INFER_VENV:-$HOME/venvs/infer}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$repo/requirements.txt"

"$VENV/bin/python" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.ones(4, device="cuda", dtype=torch.bfloat16)
    print("bf16 on gpu ok:", float((x * 2).sum()) == 8.0)
EOF

echo "activate with: source $VENV/bin/activate"
