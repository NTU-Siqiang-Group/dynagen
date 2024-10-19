FLEXGEN_PATH=$PWD/../flexgen
# for SCHEME in "original" "int4" "h2o" "dynagen"
for SCHEME in "dynagen"
do
  rm $FLEXGEN_PATH/flexgen/flex_llama.py
  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
  
  ln -s ../$SCHEME/flex_llama.py $FLEXGEN_PATH/flexgen/flex_llama.py
  ln -s ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
  ln -s ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  
  # for MODEL in "meta-llama/Llama-2-13b-chat-hf"
  for MODEL in "THUDM/LongWriter-llama3.1-8b"
  # for MODEL in "meta-llama/Meta-Llama-3.1-8B"
  # for MODEL in "NurtureAI/Meta-Llama-3-8B-Instruct-32k"
  # for MODEL in "princeton-nlp/Llama-3-8B-ProLong-64k-Base" 
  do
    CMD="--model $MODEL"
    CMD=$CMD" --percent 40 60 100 0 100 0"
    CMD=$CMD" --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 32 --gen-len 4096"
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
