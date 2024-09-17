# Speedup Evaluation
This directory contains source codes for evaluating the speedup. You can reproduce inference latency results in the paper. Some of the codes are referenced from FlexGen (ICML'23) GitHub repository.
- Getting Started (10 minutes)
- Run Experiments (7 hours)

## Getting Started (10 minutes)
```sh
sh install.sh
export CUDA_HOME=/path/to/cuda
```
For a "Hello world"-sized example, please run the following command (10 minutes):
```
python -m flexgen.flex_opt --model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 2 --num-gpu-batches 1 --prompt-len 384 --gen-len 128 --warmup-input-path flexgen/pg19_firstbook.txt --test-input-path flexgen/pg19_firstbook.txt --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 102
```
## Run Experiments (7 hours)
We provide scripts to reproduce the experiment results from Figure 14 to Figure 17. To reproduce all the results at once, please run the following commands (7 hours).
```
cd scripts
sh run_all.sh
```
If you want to reproduce the results for a specific figure, please `sh run.sh` in each corresponding directory. For example,
```
cd scripts/figure14
sh run.sh
```
Following is the amount of time to run each experiments on our system (NVIDIA RTX A6000 GPU with 48GB of memory, Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory, PCIe 3.0 x16 interconnection).
- Figure 14: 80 minutes
- Figure 15: 220 minutes
- Figure 16a: 30 minutes
- Figure 16b: 60 minutes
- Figure 17a: 20 minutes
- Figure 17b: 10 minutes
## Evaluate Accuracy (13 minutes)
We provide scripts to evaluate the accuracy of different speedup schemes using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

To begin with, please ensure that you have walked through the **Getting Started** section and installed required modules such as `lm-eval` and `ftfy`.

For a "Hello world"-sized example, please run the following command to prepare datasets (3 minutes):
```sh
$ pwd
/path/to/infinigen/speedup

$ mkdir -p ./flexgen/flexgen/results

$ cd ../accuracy/lm_eval/
$ mkdir results
$ python -u generate_task_data.py --output-file results/openbookqa-5.jsonl --task-name openbookqa --num-fewshot 5

$ cp results/openbookqa-5.jsonl ../../speedup/flexgen/flexgen/results/
$ cd ../../speedup/
```

Then run the following command to start the evaluation (10 minutes):
```sh
cd scripts
bash run_lm_eval_harness.sh openbookqa opt-6.7b ../pg19_firstbook.txt 5 0.2 4 1.0 0.2
```