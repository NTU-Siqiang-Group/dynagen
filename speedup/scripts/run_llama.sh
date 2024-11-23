FLEXGEN_PATH=$PWD/../flexgen
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-2-13b-hf"
# for MODEL in "meta-llama/Llama-3.1-70B-Instruct" 
# for MODEL in "facebook/opt-6.7B"
CMD="--model $MODEL"
CMD=$CMD" --percent 20 80 50 50 100 0"
CMD=$CMD" --gpu-batch-size 4 --num-gpu-batches 4 --prompt-len 1600 --gen-len 16 --cpu-cache-compute --computation-policy default" 
python -m flexgen.flex_llama $CMD
