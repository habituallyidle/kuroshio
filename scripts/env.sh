# Source this for development: puts the project-local CUDA toolchain (pip-installed, no root) and Rust on PATH.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$ROOT/.toolchain/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$ROOT/.toolchain/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export INKLING_WEIGHTS="${INKLING_WEIGHTS:-$HOME/vllm/hf-cache/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7}"
