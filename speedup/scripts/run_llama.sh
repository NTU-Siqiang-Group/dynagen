FLEXGEN_PATH=$PWD/../flexgen
# for SCHEME in "original" "int4" "h2o" "infinigen"
for SCHEME in "original"
do
  rm $FLEXGEN_PATH/flexgen/flex_llama.py
  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
  
  ln -s ../$SCHEME/flex_llama.py $FLEXGEN_PATH/flexgen/flex_llama.py
  ln -s ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
  ln -s ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  
  # for MODEL in "meta-llama/Llama-2-7b-chat-hf"
  # for MODEL in "THUDM/LongWriter-llama3.1-8b"
  for MODEL in "meta-llama/Meta-Llama-3.1-8B-Instruct"
  do
    CMD="--model $MODEL"
    CMD=$CMD" --percent 100 0 100 0 100 0"
    CMD=$CMD" --overlap false --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 32 --gen-len 128"
    if [ "$SCHEME" = "int4" ]
    then
      CMD=$CMD" --compress-cache"
    elif [ "$SCHEME" = "h2o" ]
    then
      CMD=$CMD" --max-num-kv 409 --hh-ratio 0.1 --hh-all"
    elif [ "$SCHEME" = "infinigen" ]
    then
      CMD=$CMD" --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 409"
    fi
    python -m flexgen.flex_llama $CMD
  done
done
