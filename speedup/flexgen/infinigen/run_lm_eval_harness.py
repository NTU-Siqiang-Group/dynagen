import argparse
import json
import os

import torch
import tqdm
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.flex_opt import OptLM, Policy, add_parser_arguments, get_opt_config
from flexgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexgen.timer import timers
from flexgen.utils import ExecutionEnv


def get_inputs(tokenizer, prompt, prompt_len = None):
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return (input_ids[0][:prompt_len],)


def init_flexgen(args, tokenizer):
    with open(args.warmup_input_path, "r") as f:
        prompt = [f.read()]
    warmup_inputs = get_inputs(tokenizer, prompt, 2048)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    # cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    # hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    model = OptLM(opt_config, env, args.path, policy, args.partial_weight_ratio, args.alpha, args.max_num_kv)
    model.generate(warmup_inputs, max_new_tokens=1, warmup=True)
    return model, env


def run_flexgen(model, tokenizer, prompt):
    # Task and policy
    inputs = get_inputs(tokenizer, prompt)

    timers("generate").reset()
    logits = model.generate(inputs, max_new_tokens=1, cut_gen_len=1, evaluate=True)
    # costs = timers("generate").costs
    
    return logits


if __name__ == "__main__":
    # Set the current working directory to the location of the Python file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    add_parser_arguments(parser)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    model, env = init_flexgen(args, tokenizer)

    requests = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() != "":
                requests.append(json.loads(line))

    results = []
    density = []
    with torch.no_grad():
        try:
            for request in tqdm.tqdm(requests):
                result = {"request": request, "result": {}}
                prompt = [request["prompt"]]
                
                input_ids = tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                ).input_ids
                logits = run_flexgen(model, tokenizer, prompt).log_softmax(dim=-1)

                values, indices = logits.squeeze(0).topk(dim=-1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

                gold_indices = input_ids[:, 1:]  # skip first
                logprobs = [None] + torch.gather(
                    logits, -1, gold_indices.unsqueeze(-1)
                ).squeeze(-1).squeeze(0).detach().cpu().tolist()
                top_logprobs = [None] + [
                    {tokenizer.convert_ids_to_tokens(i.item()): v.item()}
                    for v, i in zip(values.squeeze(-1), indices.squeeze(-1))
                ]

                result["result"] = {
                    "choices": [
                        {
                            "text": prompt,
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": logprobs,
                                "top_logprobs": top_logprobs,
                                "text_offset": [],
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "request_time": {"batch_time": 0, "batch_size": 1},
                }

                results.append(result)
        finally:
            env.close_copy_threads()

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")