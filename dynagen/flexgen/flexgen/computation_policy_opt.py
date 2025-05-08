from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from flexgen.computation_policy_interface import *
from flexgen.optimize.dynagen_optimize import DynagenOptWorksetHeuristic
from flexgen.optimize.network_config import ProfilerConfig, Llama70BConfig
from flexgen.timer import timers


class MultiStreamBase:
    def __init__(self, size, evaluate=False):
        self.size = size
        self.streams = [torch.cuda.Stream() for _ in range(size)]
        self.evaluate = evaluate
        self.executors = ThreadPoolExecutor(max_workers=size)
        self.execute_idx = 0

    def run(self, need_sync, func, *args):
        use_stream = self.streams[self.execute_idx]
        self.execute_idx = (self.execute_idx + 1) % self.size

        def _run_func():
            with torch.cuda.stream(use_stream), torch.cuda.nvtx.range(f"{func.__name__}{args}"):
                func(*args)
            return use_stream if need_sync else None

        return self.executors.submit(_run_func)


def wait_stream_finish(f):
    stream = f.result()
    if stream is not None:
        # print("Synchronizing stream")
        stream.synchronize()


class CacheLoaderManager(MultiStreamBase):
    def __init__(self, size):
        super().__init__(size)

    def load_cache(self, need_sync, func, *args):
        return self.run(need_sync, func, *args)


class ComputationStreamAlterManager(MultiStreamBase):
    def __init__(self, size):
        super().__init__(size)

    def compute(self, need_sync, func, *args):
        return self.run(need_sync, func, *args)


class ComputationPolicyOptimize(ComputationPolicyInterface):
    def generation_loop_normal(self, this, evaluate):
        raise NotImplementedError()

    def generation_loop_overlap_single_batch(self, this, evaluate):
        def load_layer_weight(i, j):
            this.load_weight(i, j, 0, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, layers_weights_sync, layers_cache_sync):
            wait_stream_finish(layers_weights_sync[j])
            layers_weights_sync[j] = None
            if this.layers[j].need_cache:
                wait_stream_finish(layers_cache_sync[j])
            layers_cache_sync[j] = None
            this.load_hidden(i, j, 0)
            this.compute_layer(i, j, 0, cpu_delegation=0)
            if j == this.num_layers - 1:
                this.sync()
            this.store_cache(i, j - 1, 0)
            this.store_hidden(i, j, 0)

        layers_weights_sync = [None for _ in range(this.num_layers)]
        layers_cache_sync = [None for _ in range(this.num_layers)]
        f = this.cache_loader.load_cache(True, load_layer_weight, 0, 0)
        layers_weights_sync[0] = f
        this.sync()
        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()
            this.update_attention_mask(i, 0)
            for j in range(this.num_layers):
                loading_weights = sum(x is not None for x in layers_weights_sync)
                loading_caches = sum(x is not None for x in layers_cache_sync)
                step = j + 2 if i == 0 else j + 10
                for l in range(j + 1, step):
                    layer = l
                    token = i
                    if layer >= this.num_layers:
                        layer = layer - this.num_layers
                        token = i + 1
                    if token >= this.execute_gen_len:
                        continue
                    if layers_weights_sync[layer] is None and loading_weights <= 1:
                        f = this.cache_loader.load_cache(True, load_layer_weight, token, layer)
                        layers_weights_sync[layer] = f
                    if layers_cache_sync[layer] is None and loading_caches <= 1:
                        f = this.cache_loader.load_cache(True, load_layer_cache, token, layer, 0, 1)
                        layers_cache_sync[layer] = f

                compute_layer(i, j, layers_weights_sync, layers_cache_sync)
            if i == 0:
                this.sync()

            timers("generate").stop()

    def generation_loop_overlap_multi_batch(self, this, evaluate):
        def load_layer_weight(i, j, k):
            this.load_weight(i, j, k, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, k, layers_weights_sync, layers_cache_sync, cpu_del):
            torch.cuda.nvtx.range_push(f"Pre-compute Sync {i}, {j}, {k}")
            wait_stream_finish(layers_weights_sync[k][j])
            layers_weights_sync[k][j] = None
            if i != 0 and this.layers[j].need_cache:
                wait_stream_finish(layers_cache_sync[k][j])
            layers_cache_sync[k][j] = None
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"Compute {i}, {j}, {k}")
            this.store_hidden(i, j, k - 1)
            this.load_hidden(i, j, k + 1)
            this.compute_layer(i, j, k, cpu_delegation=cpu_del[(i, j, k)])
            this.store_cache(i, j, k - 1, overlap=False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"Post-compute Sync {i}, {j}, {k}")
            this.sync()
            torch.cuda.nvtx.range_pop()

        optimizer = DynagenOptWorksetHeuristic(
          this.num_layers,
          this.policy.gpu_batch_size,
          this.num_gpu_batches,
          this.prompt_len,
          this.execute_gen_len,
          this.gpu_memory_capacity,
          Llama70BConfig()
        )
        optimizer.optimize()
        cache_prefetch, weight_prefetch, cpu_delegation = optimizer.get_policy()

        layers_weights_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
        layers_cache_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
        w = this.cache_loader.load_cache(True, load_layer_weight, 0, 0, 0)
        layers_weights_sync[0][0] = w
        this.load_hidden(0, 0, 0)
        this.sync()
        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()

            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)

            for j in range(this.num_layers):
                for k in range(this.num_gpu_batches):
                    torch.cuda.nvtx.range_push(f"Token {i}, Layer {j}, Batch {k}")
                    cache_prefetches = []
                    if (i, j, k) in cache_prefetch:
                        cache_prefetches = cache_prefetch[(i, j, k)]
                    weight_prefetches = []
                    if (i, j, k) in weight_prefetch:
                        weight_prefetches = weight_prefetch[(i, j, k)]
                    for token, layer, batch in cache_prefetches:
                        f = this.cache_loader.load_cache(True, load_layer_cache, token, layer, batch, cpu_delegation[(token, layer, batch)])
                        layers_cache_sync[batch][layer] = f
                    for token, layer, batch in weight_prefetches:
                        f = this.cache_loader.load_cache(True, load_layer_weight, token, layer, batch)
                        layers_weights_sync[batch][layer] = f

                    compute_layer(
                        i,
                        j,
                        k,
                        layers_weights_sync,
                        layers_cache_sync,
                        cpu_delegation
                    )
                    torch.cuda.nvtx.range_pop()
            
            if i == 0:
                this.sync()
            timers("generate").stop()

    def generation_loop_debug_multi_batch(self, this):
        def is_attn_layer(j):
            return j % 2 == 1 and j != this.num_layers - 1

        def is_mlp_layer(j):
            return j % 2 == 0 and j != 0

        def is_valid_step(i, j, k):
            if k >= this.num_gpu_batches:
                k = k % this.num_gpu_batches
                j += 1
            elif k < 0:
                k = this.num_gpu_batches - (-k % this.num_gpu_batches)
                j -= 1
            if j >= this.num_layers:
                j = j % this.num_layers
                i += 1
            elif j < 0:
                j = this.num_layers - (-j % this.num_layers)
                i -= 1
            return i >= 0 and i < this.execute_gen_len

        timers("load_weight").reset()
        timers("load_cache").reset()
        timers("store_cache").reset()
        timers("load_hidden").reset()
        timers("store_hidden").reset()
        timers("compute_cache_gpu").reset()
        timers("compute_cache_cpu").reset()
        timers("compute_mlp").reset()

        load_weight_steps = set()
        load_cache_steps = set()
        store_cache_steps = set()

        load_weight_start = torch.cuda.Event(enable_timing=True)
        load_weight_end = torch.cuda.Event(enable_timing=True)
        load_cache_start = torch.cuda.Event(enable_timing=True)
        load_cache_end = torch.cuda.Event(enable_timing=True)
        store_hidden_start = torch.cuda.Event(enable_timing=True)
        store_hidden_end = torch.cuda.Event(enable_timing=True)
        load_hidden_start = torch.cuda.Event(enable_timing=True)
        load_hidden_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        store_cache_start = torch.cuda.Event(enable_timing=True)
        store_cache_end = torch.cuda.Event(enable_timing=True)

        n = this.execute_gen_len * this.num_layers * this.num_gpu_batches
        pbar = tqdm(total=n)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(this.num_gpu_batches):
            this.load_weight(0, 0, k)
        this.load_hidden(0, 0, 0)
        this.sync()

        # Generate
        c = 1
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("prefill").start()

            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)
            for j in range(this.num_layers):
                is_attn, is_mlp = is_attn_layer(j), is_mlp_layer(j)
                if i > 0:
                    timers("decoding_gpu_batch").start()
                for k in range(this.num_gpu_batches):
                    cpu_del = k % 2 == 0

                    load_weight_start.record(stream=this.load_weight_stream)
                    this.load_weight(i, j + 1, k)
                    load_weight_end.record(stream=this.load_weight_stream)

                    load_cache_start.record(stream=this.load_cache_stream)
                    this.load_cache_dyn(i, j, k + 1, load_to_cpu=cpu_del, overlap=True)
                    load_cache_end.record(stream=this.load_cache_stream)

                    store_hidden_start.record()
                    this.store_hidden(i, j, k - 1)
                    store_hidden_end.record()

                    load_hidden_start.record()
                    this.load_hidden(i, j, k + 1)
                    load_hidden_end.record()

                    compute_start.record()
                    this.compute_layer(i, j, k, cpu_del)
                    compute_end.record()

                    store_cache_start.record(stream=this.store_cache_stream)
                    this.store_cache(i, j, k - 1)
                    store_cache_end.record(stream=this.store_cache_stream)

                    this.sync()

                    if is_valid_step(i, j + 1, k):
                        load_weight_steps.add(c)
                        timers("load_weight").costs.append(
                            load_weight_start.elapsed_time(load_weight_end) / 1000
                        )

                    if is_valid_step(i, j, k + 1) and i != 0 and is_attn and not cpu_del:
                        load_cache_steps.add(c)
                        timers("load_cache").costs.append(
                            load_cache_start.elapsed_time(load_cache_end) / 1000
                        )

                    if is_valid_step(i, j, k - 1):
                        timers("store_hidden").costs.append(
                            store_hidden_start.elapsed_time(store_hidden_end) / 1000
                        )

                    if is_valid_step(i, j, k + 1):
                        timers("load_hidden").costs.append(
                            load_hidden_start.elapsed_time(load_hidden_end) / 1000
                        )

                    if is_attn:
                        if cpu_del:
                            timers("compute_cache_cpu").costs.append(
                                compute_start.elapsed_time(compute_end) / 1000
                            )
                        else:
                            timers("compute_cache_gpu").costs.append(
                                compute_start.elapsed_time(compute_end) / 1000
                            )
                    elif is_mlp:
                        timers("compute_mlp").costs.append(
                            compute_start.elapsed_time(compute_end) / 1000
                        )

                    if is_valid_step(i, j, k - 1) and i < this.execute_gen_len - 1 and is_attn:
                        store_cache_steps.add(c)
                        timers("store_cache").costs.append(
                            store_cache_start.elapsed_time(store_cache_end) / 1000
                        )

                    pbar.update(1)
                    c += 1
                if i > 0:
                    timers("decoding_gpu_batch").stop()
            if i == 0:
                timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(this.num_layers * batch_cost)

        # Compute average cost for each step
        profiler = Llama70BConfig()
        weight_sizes = profiler.get_weights()

        def get_compute_weight_size(c, weight_gpu_percent):
            if c > n:
                return 0
            j = ((c - 1) % (this.num_layers * this.num_gpu_batches)) // this.num_gpu_batches
            return weight_sizes[j] * (100 - weight_gpu_percent) // 100

        def get_compute_cache_size(c, cache_gpu_percent):
            i = (c - 1) // (this.num_layers * this.num_gpu_batches)
            return profiler.get_cache_size(this.policy.gpu_batch_size, this.prompt_len + i) * (100 - cache_gpu_percent) // 100

        timers("htod").reset()
        timers("dtoh").reset()
        w = lc = sc = k = 0
        for cur_idx in range(1, n + 1):
            if cur_idx in load_weight_steps:
                timers("htod").costs.append(
                    timers("load_weight").costs[w] / get_compute_weight_size(cur_idx + this.num_gpu_batches, this.policy.w_gpu_percent)
                )
                w += 1
            if cur_idx in load_cache_steps:
                timers("htod").costs.append(
                    timers("load_cache").costs[lc] / get_compute_cache_size(cur_idx + 1, this.policy.cache_gpu_percent)
                )
                lc += 1
            if cur_idx in store_cache_steps:
                if k % 2 == 0:
                    timers("dtoh").costs.append(
                        timers("store_cache").costs[sc] / profiler.get_cache_size(this.policy.gpu_batch_size, 1)
                    )
                else:
                    timers("dtoh").costs.append(
                        timers("store_cache").costs[sc] / get_compute_cache_size(cur_idx - 1, this.policy.cache_gpu_percent)
                    )
                sc += 1
            k += 1
            if k >= this.num_gpu_batches:
                k = 0

        ProfilerConfig.htod_cost = timers("htod").elapsed()
        ProfilerConfig.dtoh_cost = timers("dtoh").elapsed()
        ProfilerConfig.compute_cache_gpu = timers("compute_cache_gpu").elapsed()
        ProfilerConfig.compute_cache_cpu = timers("compute_cache_cpu").elapsed()
        ProfilerConfig.compute_mlp_gpu = timers("compute_mlp").elapsed()

        # TODO: load_hidden and store_hidden
