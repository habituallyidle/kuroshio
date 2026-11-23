#!/usr/bin/env bash
# Build kuroshio and put it on your PATH. No root needed.
#   ./install.sh [--model DIR] [--prefix DIR]
#     --model DIR    the Inkling 2 Mini snapshot directory (remembered in ~/.config/kuroshio/model)
#     --prefix DIR   where to link the binary (default ~/.local/bin)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$HOME/.local/bin"; MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }

# 1. GPU: the kernels use Blackwell-only tensor-core instructions (compute capability 12.0).
if command -v nvidia-smi >/dev/null; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case "$GPU" in *"RTX 50"*|*"RTX PRO 6000 Blackwell"*) ;; *) echo "warning: found '$GPU'. kuroshio is written for the RTX 5090 (it needs a Blackwell GPU and about 21 GiB of free memory)." ;; esac
else
  echo "warning: nvidia-smi not found; is the NVIDIA driver installed?"
fi

# 2. Rust
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null; then
  echo "Rust is not installed. Install it (no root needed) and run this script again:"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  exit 1
fi

# 3. CUDA compiler. Use $CUDA_HOME or /usr/local/cuda when present; otherwise fetch NVIDIA's pip packages into
#    .toolchain/ (about 1.3 GB).
if [ -z "${CUDA_HOME:-}" ] && [ ! -x /usr/local/cuda/bin/nvcc ] && ! ls "$ROOT"/.toolchain/lib/python3*/site-packages/nvidia/cu13/bin/nvcc >/dev/null 2>&1; then
  say "Fetching the CUDA 13 compiler and runtime into .toolchain/ (pip packages, no root)"
  python3 -m venv "$ROOT/.toolchain"
  "$ROOT/.toolchain/bin/pip" install --quiet --upgrade pip
  "$ROOT/.toolchain/bin/pip" install --quiet "nvidia-cuda-nvcc>=13,<14" "nvidia-cuda-runtime>=13,<14" "nvidia-cuda-cccl>=13,<14" "nvidia-cuda-crt>=13,<14" "nvidia-nvvm>=13,<14"
fi

# 4. Build
say "Building (about a minute)"
(cd "$ROOT" && cargo build --release)

# 5. Link the binary and remember the model
mkdir -p "$PREFIX"
ln -sf "$ROOT/target/release/kuroshio" "$PREFIX/kuroshio"
if [ -n "$MODEL" ]; then
  [ -f "$MODEL/model-00001-of-00001.safetensors" ] || { echo "$MODEL does not look like the model snapshot (no model-00001-of-00001.safetensors)"; exit 1; }
  mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/kuroshio"
  (cd "$MODEL" && pwd) > "${XDG_CONFIG_HOME:-$HOME/.config}/kuroshio/model"
fi

say "Installed: $PREFIX/kuroshio"
case ":$PATH:" in *":$PREFIX:"*) ;; *) echo "note: $PREFIX is not on your PATH" ;; esac
echo "Next:  kuroshio serve          (model not downloaded yet?  hf download thinkingmachines/Inkling-2-Mini)"
