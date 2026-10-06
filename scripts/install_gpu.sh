#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 in a dedicated virtual environment"'
"$PYTHON" scripts/bootstrap_runtime.py
"$PYTHON" -m pip install 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' 'torchcodec==0.16.0' --index-url https://download.pytorch.org/whl/cu129
"$PYTHON" -m pip install 'triton==3.6.0' setuptools wheel ninja
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu129
export VLLM_PRECOMPILED_WHEEL_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0
export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}"
export VLLM_VERSION_OVERRIDE=0.1.dev18960+g6267ca0cf
"$PYTHON" -m pip install -e .runtime/sources/EasySteer/vllm-steer
"$PYTHON" -m pip install -e .runtime/sources/EasySteer 'transformers==5.16.1' gguf
"$PYTHON" -m pip install -e '.[data,test]'
"$PYTHON" scripts/bootstrap_runtime.py --verify-only
"$PYTHON" -m pip check
