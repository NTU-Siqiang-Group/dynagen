FLEXGEN_PATH=$PWD/../flexgen
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-2-13b-hf"
# MODEL="codellama/CodeLlama-34b-hf" 
# for MODEL in "facebook/opt-6.7B"
CMD="--model $MODEL"
CMD=$CMD" --percent 60 40 0 100 100 0"
CMD=$CMD" --gpu-batch-size 8 --num-gpu-batches 8 --prompt-len 1024 --gen-len 64 --computation-policy optimize --cpu-cache-compute --gpu-mem 20"
python -m flexgen.flex_llama $CMD