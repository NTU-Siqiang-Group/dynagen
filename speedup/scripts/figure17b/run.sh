FLEXGEN_PATH=$PWD/../../flexgen
rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

for PARTIAL_WEIGHT_RATIO in 0.3
do
  CMD="--model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 32 --gen-len 32 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
  CMD=$CMD" --alpha 4 --partial-weight-ratio $PARTIAL_WEIGHT_RATIO --max-num-kv 409"
  python -m flexgen.flex_opt $CMD
done
