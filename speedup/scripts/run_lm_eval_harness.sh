task=$1
shots=$4
model=$2
warmup=$3
partial_weight=$5
alpha=$6
capacity=$7
budget=$8
no_skewing=${9}

FLEXGEN_PATH=$PWD/../flexgen
for SCHEME in "original" "h2o" "infinigen" "rl_cache"
do
  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
  rm $FLEXGEN_PATH/flexgen/run_lm_eval_harness.py
  ln -s $FLEXGEN_PATH/$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
  ln -s $FLEXGEN_PATH/$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  ln -s $FLEXGEN_PATH/$SCHEME/run_lm_eval_harness.py $FLEXGEN_PATH/flexgen/run_lm_eval_harness.py

  CMD="--input-path results/${task}-${shots}.jsonl --output-path results/${task}-${shots}-${model}-${SCHEME}.jsonl --model ${model} --percent 100 0 100 0 100 0 --warmup-input-path ${warmup}"
  if [ "$SCHEME" = "infinigen" ]
  then
    CMD=$CMD" --partial-weight-ratio ${partial_weight} --alpha ${alpha}"
  elif [ "$SCHEME" = "h2o" ]
  then
    CMD=$CMD" --max-num-kv 409 --hh-ratio 0.1 --hh-all"
  fi
  python -u $FLEXGEN_PATH/flexgen/run_lm_eval_harness.py $CMD
done
