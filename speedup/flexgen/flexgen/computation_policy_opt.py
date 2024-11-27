from flexgen.computation_policy_interface import *
from flexgen.timer import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from flexgen.optimize.dynagen_optimize import DynagenOpt


class MultiStreamBase:
    def __init__(self, size):
        self.size = size
        self.streams = [torch.cuda.Stream() for _ in range(size)]
        self.executors = ThreadPoolExecutor(max_workers=size)
        self.execute_idx = 0

    def run(self, need_sync, func, *args):
        use_stream = self.streams[self.execute_idx]
        self.execute_idx = (self.execute_idx + 1) % self.size

        def _run_func():
            with torch.cuda.stream(use_stream):
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

    def generation_loop_overlap_single_batch(self, this, evaluate, profile_dir):
        def load_layer_weight(i, j, load_to_cpu=False):
            this.load_weight(i, j, 0, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, weight_handle, cache_handle):
            wait_stream_finish(weight_handle)
            # this.load_weight_stream.synchronize()
            if this.layers[j].need_cache:
                wait_stream_finish(cache_handle)

            cpu_del = this.cpu_del[j]
            this.load_hidden(i, j, 0)
            this.compute_layer(i, j, 0, cpu_delegation=cpu_del)
            this.store_cache(i, j - 1, 0)
            this.store_hidden(i, j, 0)
            # this.pop_weight(i, j, 0)

        layers_weights_sync = [None for _ in range(this.num_layers * 2)]
        layers_cache_sync = [None for _ in range(this.num_layers * 2)]
        f = this.cache_loader.load_cache(True, load_layer_weight, 0, 0, this.cpu_del[0])
        layers_weights_sync[0] = f
        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()
            this.update_attention_mask(i, 0)
            for j in range(this.num_layers):
                # load weight and cache
                for k in range(j + 1, j + 6):
                    if layers_weights_sync[k] is None:
                        f = this.cache_loader.load_cache(
                            True, load_layer_weight, i, k, this.cpu_del[k % this.num_layers]
                        )
                        layers_weights_sync[k] = f
                for k in range(j + 1, min(j + 4, this.num_layers + 5)):
                    if layers_cache_sync[k] is None:
                        f = this.cache_loader.load_cache(
                            True, load_layer_cache, i, k, 0, this.cpu_del[k % this.num_layers]
                        )
                        layers_cache_sync[k] = f
                compute_layer(i, j, layers_weights_sync[j], layers_cache_sync[j])
                if i == 0:
                    this.sync()
                if j == this.num_layers - 1:
                    layers_weights_sync = layers_weights_sync[this.num_layers :] + [
                        None for _ in range(this.num_layers)
                    ]
                    layers_cache_sync = layers_cache_sync[this.num_layers :] + [None for _ in range(this.num_layers)]

            timers("generate").stop()

    def generation_loop_overlap_multi_batch(self, this, profile_dir):
        def load_layer_weight(i, j, k):
            this.load_weight(i, j, k, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, k, weight_handle, cache_handle, layers_weights_sync, layers_cache_sync):
            wait_stream_finish(weight_handle)
            layers_weights_sync[k][j] = None
            if this.layers[j].need_cache:
                wait_stream_finish(cache_handle)
            layers_cache_sync[k][j] = None
            cpu_del = k % 2 == 0
            this.store_hidden(i, j, k - 1)
            this.load_hidden(i, j, k + 1)
            this.compute_layer(i, j, k, cpu_delegation=cpu_del)
            this.store_cache(i, j, k - 1)

        optimizer = DynagenOpt(this.num_layers, 4, this.num_gpu_batches, 512, this.execute_gen_len)
        optimizer.optimize_alter_v2()
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
                    cache_prefetches = []
                    if (i, j, k) in cache_prefetch:
                        cache_prefetches = cache_prefetch[(i, j, k)]
                    weight_prefetches = []
                    if (i, j, k) in weight_prefetch:
                        weight_prefetches = weight_prefetch[(i, j, k)]
                    for (token, layer, batch) in cache_prefetches:
                        cpu_del = cpu_delegation[(token, layer, batch)]
                        f = this.cache_loader.load_cache(True, load_layer_cache, token, layer, batch, cpu_del)
                        layers_cache_sync[batch][layer] = f
                    for (token, layer, batch) in weight_prefetches:
                        f = this.cache_loader.load_cache(True, load_layer_weight, token, layer, batch)
                        layers_weights_sync[batch][layer] = f
                    
                    compute_layer(
                        i,
                        j,
                        k,
                        layers_weights_sync[k][j],
                        layers_cache_sync[k][j],
                        layers_weights_sync,
                        layers_cache_sync,
                    )

                    if i == 0:
                        this.sync()


            timers("generate").stop()