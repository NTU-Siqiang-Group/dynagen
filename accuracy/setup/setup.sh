#!/bin/bash

LLAMA_PATH="THUDM/LongWriter-llama3.1-8b"
LLAMA_MODEL="LongWriter-llama3.1-8b"

CWD=${PWD}
cd ../transformers/src/transformers/models

mv llama/modeling_llama.py llama/modeling_llama_orig.py

cd ${CWD}
# ========= InfiniGen ============

# generate skewing matrices for llama
python gen_llama_skewing_matrix.py \
  --model "${LLAMA_PATH}" \
  --output "./skewing_matrix" 



# generate partial weight matrices for prediction
PARTIAL_RATIO=0.2

# llama
python gen_partial_weight.py \
  --skewing_matrix_path "./skewing_matrix/${LLAMA_MODEL}.pt" \
  --model "${LLAMA_PATH}" \
  --model_type "llama" \
  --partial_weight_ratio $PARTIAL_RATIO \
  --output "./weights"
