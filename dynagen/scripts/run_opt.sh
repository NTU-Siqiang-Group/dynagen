#!/bin/bash

for policy in "optimize"
do
  for MODEL in "facebook/opt-13b"
  do
    CMD="--model $MODEL"
    CMD=$CMD" --percent 70 30 0 100 100 0"
    CMD=$CMD" --gpu-batch-size 4 --num-gpu-batches 4 --prompt-len 512 --gen-len 16 --computation-policy $policy --cpu-cache-compute"
    python -m flexgen.flex_opt $CMD
  done
done