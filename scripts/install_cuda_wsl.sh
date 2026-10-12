#!/usr/bin/env bash
# Installs the CUDA toolkit inside WSL2 Ubuntu from NVIDIA's wsl-ubuntu apt repository.
#
# Only the toolkit is installed. The GPU driver comes from the Windows host and is exposed to
# WSL under /usr/lib/wsl/lib, so the `cuda` and `cuda-drivers` meta packages must not be
# installed here: they would pull in a Linux driver that conflicts with it.
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13-3}"
REPO=https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64
KEYRING=cuda-keyring_1.1-1_all.deb

if ! grep -qi microsoft /proc/version; then
  echo "This script is meant for WSL2. On native Linux use the distro's CUDA repository." >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

wget -q -O "$tmp/$KEYRING" "$REPO/$KEYRING"
sudo dpkg -i "$tmp/$KEYRING"
sudo apt-get update
sudo apt-get install -y "cuda-toolkit-$CUDA_VERSION"

# Put nvcc and the Nsight tools on PATH for login shells.
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh >/dev/null

/usr/local/cuda/bin/nvcc --version
