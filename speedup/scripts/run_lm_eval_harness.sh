task=$1
model=$2
warmup=$3
shots=$4
partial_weight=$5
alpha=$6
capacity=$7
budget=$8
no_skewing=${9}

FLEXGEN_PATH=$PWD/../flexgen
for SCHEME in "original" "h2o" "infinigen" "rl_cache"
for SCHEME in "original"
do
  echo "================== Evaluating scheme: $SCHEME =================="

  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
  rm $FLEXGEN_PATH/flexgen/run_lm_eval_harness.py
  ln -s $FLEXGEN_PATH/$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
  ln -s $FLEXGEN_PATH/$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
  ln -s $FLEXGEN_PATH/$SCHEME/run_lm_eval_harness.py $FLEXGEN_PATH/flexgen/run_lm_eval_harness.py

  CMD="--input-path results/${task}-${shots}.jsonl --output-path results/${task}-${shots}-${model}-${SCHEME}.jsonl --model ${model} --percent 100 0 100 0 100 0 --warmup-input-path ${warmup} --overlap false"
  if [ "$SCHEME" = "infinigen" ]
  then
    CMD=$CMD" --partial-weight-ratio ${partial_weight} --alpha ${alpha}"
  elif [ "$SCHEME" = "h2o" ]
  then
    CMD=$CMD" --max-num-kv 409 --hh-ratio 0.1 --hh-all"
  fi
  python -m flexgen.run_lm_eval_harness $CMD

  # Evaluate results
  python -u $FLEXGEN_PATH/../../accuracy/lm_eval/evaluate_task_result.py \
    --result-file $FLEXGEN_PATH/flexgen/results/${task}-${shots}-${model}-${SCHEME}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-name facebook/${model}

  rm $FLEXGEN_PATH/flexgen/results/${task}-${shots}-${model}-${SCHEME}.jsonl
done
