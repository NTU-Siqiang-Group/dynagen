FLEXGEN_PATH=$PWD/../flexgen
# for SCHEME in "original" "dynacache" "dynacache2" "dynagen"
for SCHEME in "original"
do
  for MODEL in "meta-llama/Llama-3.2-1B"
  # for MODEL in "meta-llama/Llama-3.1-8B-Instruct"
  # for MODEL in "meta-llama/Llama-3.1-70B-Instruct" 
  do
    CMD="--model $MODEL"
    CMD=$CMD" --percent 100 0 40 60 100 0"
    CMD=$CMD" --gpu-batch-size 4 --num-gpu-batches 1 --prompt-len 32 --gen-len 1024"
    if [ "$SCHEME" = "int4" ]
    then
      CMD=$CMD" --compress-cache"
    elif [ "$SCHEME" = "h2o" ]
    then
      CMD=$CMD" --max-num-kv 409 --hh-ratio 0.1 --hh-all"
    elif [ "$SCHEME" = "dynagen" ]
    then
      CMD=$CMD # " --warmup-input-path ../test.txt --test-input-path ../test.txt --alpha 10 --partial-weight-ratio 0.1 --max-num-kv 8192"
    fi
    python -m flexgen.flex_llama $CMD
  done
done
