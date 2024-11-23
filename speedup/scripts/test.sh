#!/bin/bash

for policy in "stream"
do
  for MODEL in "meta-llama/Llama-3.2-1B"
  do
    CMD="--model $MODEL"
    CMD=$CMD" --percent 100 0 0 100 100 0"
    CMD=$CMD" --gpu-batch-size 4 --num-gpu-batches 4 --prompt-len 512 --gen-len 32 --computation-policy $policy"
    python -m flexgen.flex_llama $CMD
  done
done