#!/bin/bash

function run_llama() {
    model=$1
    gpu_batch_size=$2
    num_gpu_batches=$3
    prompt_len=$4
    gen_len=$5
    computation_policy=$6
    cpu_cache_compute=$7
    if [ "$model" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
        percent="100 0 0 100 100 0"
    else
        percent="70 30 0 100 100 0"
    fi
    cmd="python3 -m flexgen.flex_llama --model $model --percent $percent --gpu-batch-size $gpu_batch_size --num-gpu-batches $num_gpu_batches --prompt-len $prompt_len --gen-len $gen_len --computation-policy $computation_policy"
    if [ "$cpu_cache_compute" = "True" ]; then
        cmd="$cmd --cpu-cache-compute"
    fi
    cmd="CUDA_VISIBLE_DEVICES=2 $cmd"
    # execute the command
    echo $cmd
    eval $cmd
}

function test_alter_codegen() {
    sed -i "s/optimizer.optimize_default()/optimizer.optimize_alter_v2()/g" speedup/flexgen/flexgen/computation_policy_opt.py
    sed -i "s/use_code_gen = False/use_code_gen = True/g" speedup/flexgen/flexgen/computation_policy_opt.py
    run_llama $1 $2 $3 $4 $5 $6 $7
}

function test_alter_origin() {
    sed -i "s/optimizer.optimize_default()/optimizer.optimize_alter_v2()/g" speedup/flexgen/flexgen/computation_policy_opt.py
    sed -i "s/use_code_gen = True/use_code_gen = False/g" speedup/flexgen/flexgen/computation_policy_opt.py
    run_llama $1 $2 $3 $4 $5 $6 $7
}

gpu_batch_size=8
num_gpu_batches=16
prompt_len=1024

for model in "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-2-13b-hf"
do
    for gen_len in 32 64 128 256 512
    do
        echo "Testing origin gen-len: $gen_len"
        test_alter_origin $model $gpu_batch_size $num_gpu_batches $prompt_len $gen_len "optimize" "True"
        echo "Testing codegen gen-len: $gen_len"
        test_alter_codegen $model $gpu_batch_size $num_gpu_batches $prompt_len $gen_len "optimize" "True"
    done
done