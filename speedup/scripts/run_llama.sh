FLEXGEN_PATH=$PWD/../flexgen
# for MODEL in "meta-llama/Llama-3.2-1B"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# for MODEL in "meta-llama/Llama-3.1-70B-Instruct" 
# for MODEL in "facebook/opt-6.7B"
CMD="--model $MODEL"
CMD=$CMD" --percent 100 0 0 100 100 0"
CMD=$CMD" --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 1600 --gen-len 64 --cpu-cache-compute --computation-policy alter_stream" 
python -m flexgen.flex_llama $CMD
