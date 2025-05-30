#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 main.py "$@"