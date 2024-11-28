FLEXGEN_PATH=$PWD/../flexgen
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-2-13b-hf"
# MODEL="meta-llama/Llama-3.1-70B-Instruct" 
# for MODEL in "facebook/opt-6.7B"
CMD="--model $MODEL"
CMD=$CMD" --percent 80 20 0 100 100 0"
CMD=$CMD" --gpu-batch-size 4 --num-gpu-batches 16 --prompt-len 1024 --gen-len 64 --cpu-cache-compute --compress-weight --computation-policy alter_v2" 
python -m flexgen.flex_llama $CMD
