# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="meta-llama/Llama-2-13b-hf"
MODEL=$1
PROMPT=$2
GEN=$3
VRAM=$4
WG=$5
WC=$6
CG=$7
CC=$8
B=$9
GBS=${10}
policy=${11}

cd $HOME/dynagen/dynagen/flexgen/flexgen
CMD="--model $MODEL"
CMD=$CMD" --percent ${WG} ${WC} ${CG} ${CC} 100 0"
CMD=$CMD" --gpu-batch-size ${GBS} --num-gpu-batches ${B} --prompt-len ${PROMPT} --gen-len ${GEN} --computation-policy $policy --cpu-cache-compute --gpu-mem ${VRAM}"
python -m flexgen.flex_llama $CMD
mv "fo-hf-gbs$GBS-ngbs$B-prompt$PROMPT-gen$GEN-percent-$WG-$WC-$CG-$CC-100-0-cpu-cache.log" \
"fo-hf-gbs$GBS-ngbs$B-prompt$PROMPT-gen$GEN-percent-$WG-$WC-$CG-$CC-100-0-workset-${VRAM}G.log"
