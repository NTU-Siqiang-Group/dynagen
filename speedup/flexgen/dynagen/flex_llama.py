"""
Usage:
python3 -m flexgen.flex_llama --model meta-llama/Llama-2-7b-chat-hf --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from transformers import AutoTokenizer
from flexgen.compression import CompressionConfig
from flexgen.llama_config import LlamaConfig, get_llama_config, download_llama_weights
from flexgen.pytorch_backend import (
    LlamaTorchDevice,
    TorchDisk,
    TorchMixedDevice,
    fix_recursive_import,
    TorchCPUWeightTensorManager,
    DeviceType,
)
from flexgen.flex_opt import (
    Policy,
    InputEmbed,
    OutputEmbed,
    SelfAttention,
    MLP,
    TransformerLayer,
    OptLM,
    get_filename,
    get_inputs,
)
from flexgen.timer import timers
from flexgen.utils import (
    ExecutionEnv,
    Task,
    GB,
    ValueHolder,
    array_1d,
    array_2d,
    array_3d,
    str2bool,
    project_decode_latency,
    write_benchmark_log,
)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


class LlamaInputEmbed(InputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def get_weight_specs(self, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim, self.config.dtype)
        path = os.path.join(path, "")
        return [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
        ]

    def init_weight(self, weight_manager, weight_home, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.init_cpu_weight(weight_specs, self.policy)

        weight_home.store([None] * len(weight_specs))

    def set_weight(self, weight_manager, weight_home, weight_read_buf, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.set_weight_home(weight_home, weight_specs, weight_read_buf, self.policy)

    def load_weight(self, weight_home, weight_read_buf, k):
        (w_token,) = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst),))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 3
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_token, donate[2]),) = weight_read_buf.pop()
        else:
            ((w_token, _),) = weight_read_buf.val

        h = self.compute.llama_input_embed(h, mask, w_token, self.config.pad_token_id, donate)
        hidden.val = h


class LlamaOutputEmbed(OutputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def get_weight_specs(self, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim, self.config.dtype)
        path = os.path.join(path, "")
        return [
            # w_ln
            ((h,), dtype, path + "norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight"),
        ]

    def init_weight(self, weight_manager, weight_home, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.init_cpu_weight(weight_specs, self.policy)

        weight_home.store([None] * len(weight_specs))

    def set_weight(self, weight_manager, weight_home, weight_read_buf, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.set_weight_home(weight_home, weight_specs, weight_read_buf, self.policy)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val
        h = self.compute.llama_output_embed(
            h,
            w_ln,
            w_token,
            self.config.rms_norm_eps,
            donate,
            do_sample=True,
            temperature=0.5,
            evaluate=self.task.evaluate,
        )
        hidden.val = h


class LlamaSelfAttention(SelfAttention):
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = self.compute.compressed_device if policy.compress_weight else self.compute
        self.attention_compute = self.env.cpu if self.policy.cpu_cache_compute else self.env.gpu

        self.task = None

    def get_weight_specs(self, path):
        h, n_head, n_kv_head, dtype = (
            self.config.input_dim,
            self.config.n_head,
            self.config.num_key_value_heads,
            self.config.dtype,
        )
        head_dim = h // n_head
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        return [
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
            # w_q
            ((h, n_head * head_dim), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((n_kv_head * head_dim, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((n_kv_head * head_dim, h), dtype, path + "self_attn.v_proj.weight"),
            # w_re
            ((head_dim // 2,), dtype, path + "self_attn.rotary_emb.inv_freq"),
            # w_o
            ((n_head * head_dim, h), dtype, path + "self_attn.o_proj.weight"),
        ]

    def init_weight(self, weight_manager, weight_home, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.init_cpu_weight(weight_specs, self.policy)
        weight_home.store(([None] * len(weight_specs)))

    def set_weight(self, weight_manager, weight_home, weight_read_buf, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.set_weight_home(weight_home, weight_specs, weight_read_buf, self.policy)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_q, w_k, w_v, w_re, w_o = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store(
                (
                    w_ln.smart_copy(dst2),
                    w_q.smart_copy(dst1),
                    w_k.smart_copy(dst1),
                    w_v.smart_copy(dst1),
                    w_re.smart_copy(dst1),
                    w_o.smart_copy(dst1),
                )
            )

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        n_head = self.config.n_head
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 10
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (
                (w_ln, donate[2]),
                (w_q, donate[3]),
                (w_k, donate[4]),
                (w_v, donate[5]),
                (w_re, donate[6]),
                (w_o, donate[7]),
            ) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_q, _), (w_k, _), (w_v, _), (w_re, _), (w_o, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            h, new_k_cache, new_v_cache = self.compute.llama_mha(
                h,
                position_ids,
                mask,
                w_ln,
                w_q,
                w_k,
                w_v,
                w_re,
                w_o,
                n_head,
                n_kv_head,
                donate,
                self.config.rms_norm_eps,
                self.policy.compress_cache,
                self.policy.comp_cache_config,
            )
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[8]), (v_cache, donate[9]) = cache_read_buf.pop()
            position_ids = torch.cumsum(mask.data, dim=1).long() * mask.data + 1
            position_ids = position_ids[:, -h.shape[1]].unsqueeze(1)
            h, new_k_cache, new_v_cache = self.compute.llama_mha_gen(
                h,
                position_ids,
                mask,
                w_ln,
                w_q,
                w_k,
                w_v,
                w_re,
                w_o,
                self.config.rms_norm_eps,
                n_head,
                n_kv_head,
                k_cache,
                v_cache,
                donate,
                self.policy.attn_sparsity,
                self.policy.compress_cache,
                self.policy.comp_cache_config,
            )

            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class LlamaMLP(MLP):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)

    def get_weight_specs(self, path):
        h, intermediate, dtype = (self.config.input_dim, self.config.intermediate_size, self.config.dtype)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        return [
            # w_ln
            ((h,), dtype, path + "post_attention_layernorm.weight"),
            # w_g
            ((intermediate, h), dtype, path + "mlp.gate_proj.weight"),
            # w_u
            ((intermediate, h), dtype, path + "mlp.up_proj.weight"),
            # w_d
            ((h, intermediate), dtype, path + "mlp.down_proj.weight"),
        ]

    def init_weight(self, weight_manager, weight_home, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.init_cpu_weight(weight_specs, self.policy)
        weight_home.store([None] * len(weight_specs))

    def set_weight(self, weight_manager, weight_home, weight_read_buf, path):
        weight_specs = self.get_weight_specs(path)
        weight_manager.set_weight_home(weight_home, weight_specs, weight_read_buf, self.policy)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_g, w_u, w_d = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store(
                (w_ln.smart_copy(dst2), w_g.smart_copy(dst1), w_u.smart_copy(dst1), w_d.smart_copy(dst1))
            )

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[1]), (w_g, donate[2]), (w_u, donate[3]), (w_d, donate[4])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_g, _), (w_u, _), (w_d, _)) = weight_read_buf.val

        h = self.compute.llama_mlp(h, w_ln, w_g, w_u, w_d, self.config.rms_norm_eps, donate)
        hidden.val = h


class LlamaTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i):
        self.attention = LlamaSelfAttention(config, env, policy, i)
        self.mlp = LlamaMLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute


class LlamaLM(OptLM):
    def __init__(self, config: Union[str, LlamaConfig], env: ExecutionEnv, path: str, policy: Policy):
        if isinstance(config, str):
            config = get_llama_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(LlamaInputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(LlamaSelfAttention(self.config, self.env, self.policy, i))
                layers.append(LlamaMLP(self.config, self.env, self.policy, i))
            else:
                layers.append(LlamaTransformerLayer(self.config, self.env, self.policy, i))
        layers.append(LlamaOutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.weight_manager = TorchCPUWeightTensorManager(self.env)
        self.init_all_weights()

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.config.name, self.config.org, self.path, self.config.hf_token)

        self.layers[j].init_weight(self.weight_manager, self.weight_home[j], expanded_path)

    def generate(
        self,
        inputs: Union[np.array, List[List[int]]],
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        stop: Optional[int] = None,
        debug_mode: Optional[str] = None,
        cut_gen_len: Optional[int] = None,
        verbose: int = 0,
        warmup: bool = False,
        evaluate: bool = False,
    ):
        if evaluate:
            assert max_new_tokens == 1 and self.num_gpu_batches == 1 and self.policy.gpu_batch_size == 1

        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            evaluate=evaluate,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        self.warmup = warmup

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len), self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        self.generation_loop_normal(evaluate)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        if evaluate:
            return self.hidden[0][-1][0].val.data.detach().cpu()

        return self.output_ids

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(
            self.hidden[i][j][k],
            self.cache_read_buf[j][k],
            self.weight_read_buf[j],
            self.attention_mask[k],
            self.cache_write_buf[j][k],
            i,
            k,
        )
    
    def generation_loop_normal(self, evaluate):
        self.set_weight()
        for i in tqdm(range(self.execute_gen_len)):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                    self.load_cache(i, j, k)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    if evaluate and j == self.num_layers - 1:
                        self.sync()
                        break
                    self.sync()
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k)
                    # if j in self.attn_layer[1:-1] and (i > 0):
                    #     self.prefetch_cache(i, j, k, overlap=True)
                    #     self.prefetch_evt.record()
            self.set_weight()
            timers("generate").stop()

def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Write a 30000-word article on the history of the Roman Empire."]
    input_ids = tokenizer(prompts, padding="max_length", max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = LlamaTorchDevice("cuda:0")
    cpu = LlamaTorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(
        args.gpu_batch_size,
        args.num_gpu_batches,
        args.percent[0],
        args.percent[1],
        args.percent[2],
        args.percent[3],
        args.percent[4],
        args.percent[5],
        args.overlap,
        args.sep_layer,
        args.pin_weight,
        args.cpu_cache_compute,
        args.attn_sparsity,
        args.compress_weight,
        CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        args.compress_cache,
        CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    llama_config = get_llama_config(args.model, hf_token=args.hf_token, pad_token_id=tokenizer.eos_token_id)
    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(
        f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
        f"cache size: {cache_size/GB:.3f} GB, "
        f"hidden size (prefill): {hidden_size/GB:.3f} GB"
    )

    print("init weight...")
    model = LlamaLM(llama_config, env, args.path, policy)

    try:
        print("warmup - generate")
        output_ids = model.generate(warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs,
            max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode,
            cut_gen_len=cut_gen_len,
            verbose=args.verbose,
        )
        print(output_ids)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * "-" + "\n"
        for i in [0]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(
        filename,
        llama_config.model_bytes(),
        cache_size,
        hidden_size,
        gpu_peak_mem,
        projected,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="The model name.")
    parser.add_argument("--hf-token", type=str, help="The huggingface token for accessing gated repo.")
    parser.add_argument(
        "--path",
        type=str,
        default="~/llama_weights",
        help="The path to the model weights. If there are no cached weights, "
        "FlexGen will automatically download them from HuggingFace.",
    )
    parser.add_argument(
        "--offload-dir", type=str, default="~/flexgen_offload_dir", help="The directory to offload tensors. "
    )
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int, help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str, choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument(
        "--percent",
        nargs="+",
        type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
        "the percentage of weight on GPU, "
        "the percentage of weight on CPU, "
        "the percentage of attention cache on GPU, "
        "the percentage of attention cache on CPU, "
        "the percentage of activations on GPU, "
        "the percentage of activations on CPU",
    )
    parser.add_argument("--sep-layer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true", help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true", help="Whether to compress cache.")
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--overlap", type=str2bool, nargs="?", const=True, default=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
