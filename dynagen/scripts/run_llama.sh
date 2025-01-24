FLEXGEN_PATH=$PWD/../flexgen
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-2-13b-hf"
# MODEL="codellama/CodeLlama-34b-hf" 
# for MODEL in "facebook/opt-6.7B"
CMD="--model $MODEL"
CMD=$CMD" --percent 20 80 0 100 100 0"
CMD=$CMD" --gpu-batch-size 16 --num-gpu-batches 1 --prompt-len 2048 --gen-len 16 --computation-policy optimize --cpu-cache-compute" 
python -m flexgen.flex_llama $CMD