from dataclasses import dataclass
from math import ceil

import numpy as np
from tqdm import tqdm
from BTrees.IIBTree import IIBTree
from BTrees.IOBTree import IOBTree
from BTrees.QOBTree import QOBTree

from flexgen.optimize.network_config import ProfilerConfig


class DynagenOpt:
    def __init__(self, num_layers, batch_size, num_gpu_batches, prompt_len, gen_len, cpu_delegation_percent, num_prefetch_weight_layers, num_prefetch_cache_batches, profiler=ProfilerConfig()):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        assert type(cpu_delegation_percent) is int and cpu_delegation_percent >= 0 and cpu_delegation_percent <= 100
        self.cpu_delegation_percent = cpu_delegation_percent
        assert type(num_prefetch_weight_layers) is int and num_prefetch_weight_layers >= 0
        self.num_prefetch_weight_layers = num_prefetch_weight_layers
        assert type(num_prefetch_cache_batches) is int and num_prefetch_cache_batches >= 0
        self.num_prefetch_cache_batches = num_prefetch_cache_batches

        self.profiler = profiler  # for one layer, placeholder
        self.weights_size = self.profiler.get_weights()

        # Assumption 2: the batch can fully saturate the GPU memory.
        # 1. Prefetch & offload policy
        #   - KV cache
        #   - Weight
        self.cache_prefetch = np.array(
            [None] * gen_len * num_layers * num_gpu_batches
        )  # at which "step" the cache is fetched
        # only (i, j, 0) is valid for weight prefetch
        self.weight_prefetch = np.array(
            [None] * gen_len * num_layers * num_gpu_batches
        )  # at which "step" the weight is fetched
        # 2. TODO: KV cache percentage and weight percentage (initial value, they are fetched into the buffer gradually
        #   according to the prefetch policy). Currently assume both of them are stored in CPU memory.
        # 3. CPU delegation
        # only (i, j, 0) is valid for cpu delegation
        self.cpu_delegation = np.array(
            [0] * gen_len * num_layers * num_gpu_batches
        )  # which batch should be submitted to
        # TODO: considering compute two batch at a time. One in GPU and one in CPU

    def get_policy(self):
        cache_prefetch = {}
        weight_prefetch = {}
        cpu_delegation = {}
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    cache_prefetch_idx = self.cache_prefetch[self._idx(i, j, k)]
                    weight_prefetch_idx = self.weight_prefetch[self._idx(i, j, k)]

                    if cache_prefetch_idx is not None:
                        if self._decode(cache_prefetch_idx) not in cache_prefetch:
                            cache_prefetch[self._decode(cache_prefetch_idx)] = []
                        cache_prefetch[self._decode(cache_prefetch_idx)].append((i, j, k))

                    if weight_prefetch_idx is not None:
                        if self._decode(weight_prefetch_idx) not in weight_prefetch:
                            weight_prefetch[self._decode(weight_prefetch_idx)] = []
                        weight_prefetch[self._decode(weight_prefetch_idx)].append((i, j, k))

                    cpu_delegation[(i, j, k)] = self.cpu_delegation[self._idx(i, j, 0)]

        return cache_prefetch, weight_prefetch, cpu_delegation

    def _idx(self, token, layer, batch):
        return token * self.num_layers * self.num_gpu_batches + layer * self.num_gpu_batches + batch

    def _decode(self, idx):
        token = idx // (self.num_layers * self.num_gpu_batches)
        layer = (idx % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        batch = idx % self.num_gpu_batches
        return (token, layer, batch)

    def optimize(self):
        layers_weights_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_cache_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_weights_sync[0][0] = 1
        skip_cpu_del = int(ceil((self.num_layers / 100.0) / (self.cpu_delegation_percent / 100.0)))
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    loading_weights = sum(x is not None for sublist in layers_weights_sync for x in sublist)
                    loading_caches = sum(x is not None for sublist in layers_cache_sync for x in sublist)
                    step = k + 2 if i == 0 else k + self.num_gpu_batches * 10
                    for l in range(k + 1, step):
                        batch = l % self.num_gpu_batches
                        layer = j + l // self.num_gpu_batches
                        token = i
                        if layer >= self.num_layers:
                            layer = layer - self.num_layers
                            token = i + 1
                        if token >= self.gen_len:
                            continue
                        if layers_weights_sync[batch][layer] is None and loading_weights <= self.num_gpu_batches * self.num_prefetch_weight_layers:
                            self.weight_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            layers_weights_sync[batch][layer] = 1
                            loading_weights += 1
                        if layers_cache_sync[batch][layer] is None and loading_caches <= self.num_prefetch_cache_batches:
                            self.cache_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            self.cpu_delegation[self._idx(token, layer, batch)] = self.cpu_delegation_percent != 0 and (j % skip_cpu_del) == 0
                            layers_cache_sync[batch][layer] = 1
                            loading_caches += 1
                    # compute
                    layers_weights_sync[k][j] = None
                    layers_cache_sync[k][j] = None


@dataclass
class PrefetchPolicy:
    io_prefetch_sequence: np.ndarray  # compute_step, prefetch_step, size, is_weight, finish
    compute_step_btree: IIBTree
    prefetch_step_btree: IOBTree
    latencies: np.ndarray
    mem_consumption: np.ndarray

    def copy_original(self):
        return PrefetchPolicy(
            self.io_prefetch_sequence.copy(),
            IIBTree(self.compute_step_btree.items()),
            IOBTree(self.prefetch_step_btree.items()),
            self.latencies.copy(),
            self.mem_consumption.copy(),
        )

    def copy(self):
        latencies = np.append(self.latencies, 0)
        mem_consumption = np.append(self.mem_consumption, self.mem_consumption[-1])
        return PrefetchPolicy(
            self.io_prefetch_sequence.copy(),
            IIBTree(self.compute_step_btree.items()),
            IOBTree(self.prefetch_step_btree.items()),
            latencies,
            mem_consumption,
        )

    def insert_weight_prefetch(self, compute_step, prefetch_step, size, io_cost):
        last_io_prefetch_idx = self.prefetch_step_btree.get(prefetch_step, None)
        last_finish = (
            0 if last_io_prefetch_idx is None else self.io_prefetch_sequence[last_io_prefetch_idx]["finish"].max()
        )
        finish = last_finish + io_cost
        self.io_prefetch_sequence = np.append(
            self.io_prefetch_sequence,
            np.array([(compute_step, prefetch_step, size, True, finish)], dtype=self.io_prefetch_sequence.dtype),
        )
        self.compute_step_btree[compute_step] = len(self.io_prefetch_sequence) - 1
        self.prefetch_step_btree[prefetch_step] = [
            *self.prefetch_step_btree.get(prefetch_step, ()),
            len(self.io_prefetch_sequence) - 1,
        ]
        self.mem_consumption[prefetch_step:] += size

        for filtered_io_prefetch_idx in self.prefetch_step_btree.values(min=prefetch_step, excludemin=True):
            self.io_prefetch_sequence["finish"][filtered_io_prefetch_idx] += io_cost

    def insert_cache_prefetch(self, compute_step, prefetch_step, size, io_cost):
        last_io_prefetch_idx = self.prefetch_step_btree.get(prefetch_step, None)
        last_finish = (
            0 if last_io_prefetch_idx is None else self.io_prefetch_sequence[last_io_prefetch_idx]["finish"].max()
        )
        finish = last_finish + io_cost
        self.io_prefetch_sequence = np.append(
            self.io_prefetch_sequence,
            np.array([(compute_step, prefetch_step, size, False, finish)], dtype=self.io_prefetch_sequence.dtype),
        )
        self.compute_step_btree[compute_step] = len(self.io_prefetch_sequence) - 1
        self.prefetch_step_btree[prefetch_step] = [
            *self.prefetch_step_btree.get(prefetch_step, ()),
            len(self.io_prefetch_sequence) - 1,
        ]
        self.mem_consumption[prefetch_step:] += size

        for filtered_io_prefetch_idx in self.prefetch_step_btree.values(min=prefetch_step, excludemin=True):
            self.io_prefetch_sequence["finish"][filtered_io_prefetch_idx] += io_cost

    def update_latencies(self, prefetch_step, compute_cost):
        for i in range(prefetch_step, len(self.latencies)):
            filtered_io_prefetch = self.io_prefetch_sequence[self.compute_step_btree.get(i, [])]
            wait_time = (
                0 if filtered_io_prefetch.size == 0 else max(0, filtered_io_prefetch["finish"] - self.latencies[i - 1])
            )
            self.latencies[i] += self.latencies[i - 1] + wait_time + compute_cost[i]

    def get_last_latency(self):
        return self.latencies[-1]

    def get_last_mem_consumption(self):
        return int(np.ceil(self.mem_consumption[-1] / (1 << 30)))


class DynagenOptBruteforce:
    def __init__(
        self,
        num_layers,
        batch_size,
        num_gpu_batches,
        prompt_len,
        gen_len,
        gpu_memory_capacity,
        profiler=ProfilerConfig(),
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.n = gen_len * num_layers * num_gpu_batches
        self.gpu_memory_capacity = gpu_memory_capacity
        self.profiler = profiler

        self.weight_sizes = profiler.get_weights()
        self.policies = np.full(self.n + 1, None, dtype=object)
        self.policies[1] = PrefetchPolicy(
            np.array(
                [(1, 0, self.weight_sizes[0], True, 0.0)],
                dtype=[
                    ("compute_step", "i8"),
                    ("prefetch_step", "i8"),
                    ("size", "i8"),
                    ("is_weight", "?"),
                    ("finish", "f8"),
                ],
            ),
            IIBTree({1: 0}),
            IOBTree({0: (0,)}),
            np.array([0, self.profiler.get_compute_mlp_gpu()]),
            np.array([0, self.weight_sizes[0]]),
        )

        self.compute_costs = np.zeros(self.n + 1)
        i, j = 0, 0
        for c in range(1, self.n + 1, num_gpu_batches):
            if j % 2 == 0 or j == self.num_layers - 1:
                self.compute_costs[c : c + num_gpu_batches] = self.profiler.get_compute_mlp_gpu()
            else:
                self.compute_costs[c : c + num_gpu_batches] = self.profiler.get_compute_cache_gpu()
            j += 1
            if j == num_layers - 1:
                j = 0
                i += 1

    # Counts from 0
    def _decode(self, c):
        i = c // (self.num_layers * self.num_gpu_batches)
        j = (c % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        k = c % self.num_gpu_batches
        return i, j, k

    def get_weight_prefetch_bounds(self, c):
        if not self.is_weight_prefetch_valid(c):
            return range(c, c + 1)
        lower_bound = max(1, c - self.num_gpu_batches * (self.num_layers - 1))
        return range(lower_bound, c)

    def get_cache_prefetch_bounds(self, c):
        if not self.is_cache_prefetch_valid(c):
            return range(c, c + 1)
        lower_bound = max(1, c - self.num_gpu_batches * self.num_layers + 1)
        return range(lower_bound, c)

    def is_weight_prefetch_valid(self, c):
        k = (c - 1) % self.num_gpu_batches
        return k == 0 and c != 1

    def get_weight_size(self, c):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return self.weight_sizes[j]

    def is_cache_prefetch_valid(self, c):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return j % 2 == 1 and j != self.num_layers - 1

    def get_cache_size(self, c):
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        return self.profiler.get_cache_size(self.batch_size, self.prompt_len + i)

    def is_weight_offload_valid(self, c):
        k = (c - 1) % self.num_gpu_batches
        return k == 0 and c != 1

    def is_cache_offload_valid(self, c):
        _, j, k = self._decode(c - 1)
        return j % 2 == 1 and k > 0 or j > 0 and j % 2 == 0 and k == 0

    def optimize(self):
        for c in range(2, self.n + 1):
            assert self.policies[c - 1] is not None
            prev_policy = self.policies[c - 1].copy()
            for i_weight in self.get_weight_prefetch_bounds(c):
                for i_cache in self.get_cache_prefetch_bounds(c):
                    policy = prev_policy.copy_original()
                    if self.is_weight_prefetch_valid(c):
                        weight_size = self.get_weight_size(c)
                        policy.insert_weight_prefetch(
                            c, i_weight, weight_size, self.profiler.get_htod_cost(weight_size)
                        )

                    if self.is_cache_prefetch_valid(c):
                        cache_size = self.get_cache_size(c)
                        policy.insert_cache_prefetch(c, i_cache, cache_size, self.profiler.get_htod_cost(cache_size))

                    policy.update_latencies(min(i_weight, i_cache), self.compute_costs)
                    if self.is_weight_offload_valid(c):
                        policy.mem_consumption[-1] -= self.get_weight_size(c - 1)
                    if self.is_cache_offload_valid(c):
                        policy.mem_consumption[-1] -= self.get_cache_size(c - 1)
                    mem_consumption = policy.get_last_mem_consumption()
                    if mem_consumption > self.gpu_memory_capacity:
                        continue

                    if self.policies[c] is None or policy.get_last_latency() < self.policies[c].get_last_latency():
                        self.policies[c] = policy

    def get_policy(self):
        cache_prefetch = {}
        weight_prefetch = {}
        cpu_delegation = {}
        policy = self.policies[self.n]
        for compute_step, prefetch_step, _, is_weight, _ in policy.io_prefetch_sequence[1:]:
            if is_weight:
                weight_prefetch.setdefault(self._decode(prefetch_step - 1), []).append(self._decode(compute_step - 1))
            else:
                cache_prefetch.setdefault(self._decode(prefetch_step - 1), []).append(self._decode(compute_step - 1))
                cpu_delegation[self._decode(compute_step - 1)] = 1
        return cache_prefetch, weight_prefetch, cpu_delegation


class DynagenOptWorksetHeuristic:
    def __init__(
        self,
        num_layers,
        batch_size,
        num_gpu_batches,
        prompt_len,
        gen_len,
        gpu_memory_capacity,
        profiler=ProfilerConfig(),
        max_num_prefetch_batches=0,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.gpu_memory_capacity = int(gpu_memory_capacity * (1 << 30))
        self.max_num_prefetch_batches = max(
            0, min(num_layers * num_gpu_batches - 1, max_num_prefetch_batches)
        )  # clamp(max_num_prefetch_batches, 0, num_layers * num_gpu_batches - 1)
        self.profiler = profiler

        self.weight_sizes = profiler.get_weights()
        self.n = gen_len * num_layers * num_gpu_batches

        self.cpu_del = np.zeros(self.n + 1, np.uint32)
        self.weight_prefetch = np.zeros(self.n + 1, np.uint64)
        self.cache_prefetch = np.zeros(self.n + 1, np.uint64)

    # Count from 0
    def _decode(self, c):
        i = c // (self.num_layers * self.num_gpu_batches)
        j = (c % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        k = c % self.num_gpu_batches
        return int(i), int(j), int(k)

    def need_weight(self, c):
        k = (c - 1) % self.num_gpu_batches
        return k == 0 and c != 1

    def need_cache(self, c):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return j % 2 == 1 and j != self.num_layers - 1

    def prefetch_range(self, c):
        if self.max_num_prefetch_batches == 0:
            return range(c, min(c + self.num_layers * self.num_gpu_batches, self.n + 1))
        return range(c, min(c + self.max_num_prefetch_batches + 1, self.n + 1))

    def get_weight_size(self, c):
        if c > self.n:
            return 0
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return self.weight_sizes[j]

    def get_cache_size(self, c):
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        return self.profiler.get_cache_size(self.batch_size, self.prompt_len + i)

    def get_remaining_size(self, mem_consumption):
        return self.gpu_memory_capacity - mem_consumption

    def is_weight_offload_valid(self, c):
        k = (c - 1) % self.num_gpu_batches
        return k == self.num_gpu_batches - 1 and c != 1

    def is_cache_offload_valid(self, c):
        return self.need_cache(c) and self.cpu_del[c] == 0

    def optimize(self):
        def get_first_step_remaining(r, c, is_weight):
            iterator = remaining_sizes.iterkeys(min=r, max=self.gpu_memory_capacity, excludemax=True)
            while True:
                key = next(iterator)
                p, mem_consumption = remaining_sizes[key]
                if is_weight and c - p < self.num_layers * self.num_gpu_batches - 1:
                    return p, mem_consumption
                if not is_weight and c - p < self.num_layers * self.num_gpu_batches:
                    return p, mem_consumption

        mem_consumption = self.weight_sizes[0]
        remaining_sizes = QOBTree()
        weight_prefetched = np.zeros(self.n + 1, bool)
        weight_prefetched[: self.num_gpu_batches + 1] = True
        cache_prefetched = np.zeros(self.n + 1, bool)
        for i in range(1, self.n + 1):
            if not self.need_cache(i):
                cache_prefetched[i] = True

        i, p, prev = 1, 1, 1
        progress = tqdm(total=self.n)
        while i < self.n:
            prefetch_range = self.prefetch_range(p)
            for p in prefetch_range:
                weight_size = self.get_weight_size(p)
                cache_size = self.get_cache_size(p)
                if not weight_prefetched[p]:
                    if mem_consumption + weight_size > self.gpu_memory_capacity:
                        break
                    mem_consumption += weight_size
                    weight_prefetched[p : p + self.num_gpu_batches] = True
                    self.weight_prefetch[p] = i
                if cache_size > 0 and not cache_prefetched[p]:
                    if mem_consumption + cache_size > self.gpu_memory_capacity:
                        break
                    mem_consumption += cache_size
                    cache_prefetched[p] = True
                    self.cache_prefetch[p] = i
            remaining_sizes.clear()
            for c in range(i, prefetch_range.stop):
                if i != c:
                    if self.is_weight_offload_valid(c) and weight_prefetched[c]:
                        mem_consumption -= self.get_weight_size(c - 1)
                        r = self.get_remaining_size(mem_consumption)
                        if r not in remaining_sizes:
                            remaining_sizes[r] = (c, mem_consumption)
                    if self.is_cache_offload_valid(c) and cache_prefetched[c]:
                        mem_consumption -= self.get_cache_size(c - 1)
                        r = self.get_remaining_size(mem_consumption)
                        if r not in remaining_sizes:
                            remaining_sizes[r] = (c, mem_consumption)
                if not weight_prefetched[c]:
                    assert i != c, "weight for the current computing step should not be prefetched in the same step"
                    i, mem_consumption = get_first_step_remaining(weight_size, c, True)
                    break
                if not cache_prefetched[c]:
                    if i == c:
                        self.cpu_del[c] = 1
                        cache_prefetched[c] = True
                    else:
                        i, mem_consumption = get_first_step_remaining(cache_size, c, False)
                    break
            if c == prefetch_range.stop - 1:
                i = c
                mem_consumption = self.get_weight_size(i) + (self.get_cache_size(i) if self.need_cache(i) else 0)
            progress.update(i - prev)
            prev = i
        progress.update(1)

    def get_policy(self):
        cache_prefetch = {}
        weight_prefetch = {(0, 0, 0): [(0, 0, k) for k in range(1, self.num_gpu_batches)]}
        cpu_delegation = {}
        i, j, k = 0, 0, 0
        for c in range(1, self.n + 1):
            if self.cache_prefetch[c] != 0 and self.need_cache(c):
                cache_prefetch.setdefault(self._decode(self.cache_prefetch[c] - 1), []).append((i, j, k))
            if self.weight_prefetch[c] != 0 and self.need_weight(c):
                assert k == 0
                for batch in range(self.num_gpu_batches):
                    weight_prefetch.setdefault(self._decode(self.weight_prefetch[c] + batch - 1), []).append(
                        (i, j, batch)
                    )
            cpu_delegation[(i, j, k)] = self.cpu_del[c]
            k += 1
            if k == self.num_gpu_batches:
                k = 0
                j += 1
                if j == self.num_layers:
                    j = 0
                    i += 1
        return cache_prefetch, weight_prefetch, cpu_delegation
