import numpy as np
from BTrees.QOBTree import QOBTree
from tqdm import tqdm

from flexgen.optimize.network_config import ProfilerConfig


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
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return i != 0 and j % 2 == 1 and j != self.num_layers - 1

    def prefetch_range(self, c):
        if self.max_num_prefetch_batches == 0:
            return range(c, min(c + self.num_layers * self.num_gpu_batches, self.n + 1))
        return range(c, min(c + self.max_num_prefetch_batches + 1, self.n + 1))

    def get_compute_weight_size(self, c, weight_gpu_percent):
        if c > self.n:
            return 0
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return self.weight_sizes[j] * (100 - weight_gpu_percent) // 100

    def get_compute_cache_size(self, c, cache_gpu_percent):
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        return self.profiler.get_cache_size(self.batch_size, self.prompt_len + i) * (100 - cache_gpu_percent) // 100

    def get_remaining_size(self, mem_consumption):
        return self.gpu_memory_capacity - mem_consumption

    def is_weight_offload_valid(self, c):
        k = (c - 1) % self.num_gpu_batches
        return k == self.num_gpu_batches - 1 and c != 1

    def is_cache_offload_valid(self, c, cpu_del):
        return self.need_cache(c) and cpu_del[c] == 0
    
    def optimize(self):
        min_cost = float('inf')
        percents = [(w, c) for w in range(100, -1, -10) for c in range(100, -1, -10)]
        for weight_percent, cache_percent in percents:
            try:
                cost, *policy = self.optimize_policy(weight_percent=weight_percent, cache_percent=cache_percent)
                if cost < min_cost:
                    min_cost = cost
                    self.cache_prefetch, self.weight_prefetch, self.cpu_del = policy
                else:
                    return weight_percent, cache_percent
            except:
                continue

    def optimize_policy(self, position=0, weight_percent=0, cache_percent=0):
        assert type(weight_percent) is int and 0 <= weight_percent <= 100
        assert type(cache_percent) is int and 0 <= cache_percent <= 100

        def get_first_step_remaining(r, c, is_weight):
            iterator = remaining_sizes.iterkeys(min=r, max=self.gpu_memory_capacity)
            while True:
                key = next(iterator)
                p, mem_consumption = remaining_sizes[key]
                if is_weight and c - p < self.num_layers * self.num_gpu_batches - 1:
                    return p, mem_consumption
                if not is_weight and c - p < self.num_layers * self.num_gpu_batches:
                    return p, mem_consumption

        weight_prefetch = np.zeros(self.n + 1, np.int64)
        cache_prefetch = np.zeros(self.n + 1, np.int64)
        cpu_del = np.zeros(self.n + 1, np.int32)

        mem_consumption = np.sum(self.weight_sizes[1:], dtype=np.uint64) * weight_percent // 100 + self.weight_sizes[0]
        mem_consumption += self.profiler.get_cache_size(self.num_gpu_batches * self.batch_size, self.prompt_len + self.gen_len) * self.profiler.num_hidden_layers * cache_percent // 100
        mem_consumption = int(mem_consumption)
        assert mem_consumption <= self.gpu_memory_capacity, (
            "memory consumption should be less than gpu memory capacity"
        )
        remaining_sizes = QOBTree()

        weight_prefetched = np.zeros(self.n + 1, bool)
        weight_prefetched[: self.num_gpu_batches + 1] = True
        cache_prefetched = np.zeros(self.n + 1, bool)
        for i in range(1, self.n + 1):
            if not self.need_cache(i):
                cache_prefetched[i] = True

        i, p, prev = 1, 1, 1
        with tqdm(total=self.n, position=position, desc="Optimizing", leave=False) as progress:
            while i < self.n:
                prefetch_range = self.prefetch_range(p)
                for p in prefetch_range:
                    weight_size = self.get_compute_weight_size(p, weight_percent)
                    cache_size = self.get_compute_cache_size(p, cache_percent)
                    if not weight_prefetched[p]:
                        if mem_consumption + weight_size > self.gpu_memory_capacity:
                            break
                        mem_consumption += weight_size
                        weight_prefetched[p : p + self.num_gpu_batches] = True
                        weight_prefetch[p] = i
                    if cache_size > 0 and not cache_prefetched[p]:
                        if mem_consumption + cache_size > self.gpu_memory_capacity:
                            break
                        mem_consumption += cache_size
                        cache_prefetched[p] = True
                        cache_prefetch[p] = i
                remaining_sizes.clear()
                for c in range(i, prefetch_range.stop):
                    if i != c:
                        if self.is_weight_offload_valid(c) and weight_prefetched[c]:
                            mem_consumption -= self.get_compute_weight_size(c - 1, weight_percent)
                            r = self.get_remaining_size(mem_consumption)
                            if r not in remaining_sizes:
                                remaining_sizes[r] = (c, mem_consumption)
                        if self.is_cache_offload_valid(c, cpu_del) and cache_prefetched[c]:
                            mem_consumption -= self.get_compute_cache_size(c - 1, cache_percent)
                            r = self.get_remaining_size(mem_consumption)
                            if r not in remaining_sizes:
                                remaining_sizes[r] = (c, mem_consumption)
                    if not weight_prefetched[c]:
                        assert i != c, (
                            "weight for the current computing step should not be prefetched in the same step"
                        )
                        weight_size = self.get_compute_weight_size(c, weight_percent)
                        i, mem_consumption = get_first_step_remaining(weight_size, c, True)
                        break
                    if not cache_prefetched[c]:
                        if i == c:
                            cpu_del[c] = 1
                            cache_prefetched[c] = True
                        else:
                            cache_size = self.get_compute_cache_size(c, cache_percent)
                            i, mem_consumption = get_first_step_remaining(cache_size, c, False)
                        break
                if c == prefetch_range.stop - 1:
                    i = c
                    mem_consumption = self.get_compute_weight_size(i, weight_percent) + (
                        self.get_compute_cache_size(i, cache_percent) if self.need_cache(i) else 0
                    )
                progress.update(i - prev)
                prev = i
            progress.update(1)

        cost = self.get_cost_from_policy(cache_prefetch, weight_prefetch, cpu_del, weight_percent, cache_percent)
        return cost, cache_prefetch, weight_prefetch, cpu_del

    def get_policy(self):
        cache_prefetch_dict = {}
        weight_prefetch_dict = {
            (0, 0, 0): [(0, 0, k) for k in range(1, self.num_gpu_batches)]
        }
        cpu_delegation = {}
        i, j, k = 0, 0, 0
        for c in range(1, self.n + 1):
            if self.cache_prefetch[c] != 0 and self.need_cache(c):
                cache_prefetch_dict.setdefault(
                    self._decode(self.cache_prefetch[c] - 1), []
                ).append((i, j, k))
            if self.weight_prefetch[c] != 0 and self.need_weight(c):
                assert k == 0
                for batch in range(self.num_gpu_batches):
                    weight_prefetch_dict.setdefault(
                        self._decode(self.weight_prefetch[c] + batch - 1), []
                    ).append((i, j, batch))
            cpu_delegation[(i, j, k)] = self.cpu_del[c]
            k += 1
            if k == self.num_gpu_batches:
                k = 0
                j += 1
                if j == self.num_layers:
                    j = 0
                    i += 1
        return cache_prefetch_dict, weight_prefetch_dict, cpu_delegation

    def get_io_cost_from_policy(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        io_costs = np.zeros(self.n + 1)
        for cur_idx in range(1, self.n + 1):
            if self.need_weight(cur_idx):
                weight_prefetch_idx = weight_prefetch[cur_idx]
                if io_costs[weight_prefetch_idx] == 0:
                    io_costs[weight_prefetch_idx] = (
                        0 if weight_prefetch_idx == 0 else io_costs[weight_prefetch_idx - 1]
                    )
                j = (cur_idx % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
                prefetch_j = (weight_prefetch_idx % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
                if prefetch_j != j:
                    io_time = self.profiler.get_htod_cost(
                        self.get_compute_weight_size(cur_idx, weight_percent)
                    )
                    io_costs[weight_prefetch_idx] += io_time

            # store cache
            if not self.need_cache(cur_idx):
                continue
            if io_costs[cur_idx] == 0 and cur_idx != 1:
                io_costs[cur_idx] = io_costs[cur_idx - 1]

            cpu_del = cpu_delegation[cur_idx]
            if cpu_del:
                # 1. the cache is current stored in CPU, no htod cost is required
                # 2. the new k v should be transfered back to CPU (dtoh)
                io_costs[cur_idx] += self.profiler.get_dtoh_cost(
                    self.profiler.get_cache_size(self.batch_size, 1)
                )
            else:
                # 1. the cache is stored in CPU, htod cost is required
                # 2. the new k v is not required to transfer to CPU. But the whole KV cache should be transfered to CPU after computation
                cache_prefetch_idx = cache_prefetch[cur_idx]
                if io_costs[cache_prefetch_idx] == 0:
                    io_costs[cache_prefetch_idx] = (
                        0 if cache_prefetch_idx == 0 else io_costs[cache_prefetch_idx - 1]
                    )
                io_time = self.profiler.get_htod_cost(
                    self.get_compute_cache_size(cur_idx, cache_percent)
                )
                # prefetch cache
                io_costs[cache_prefetch_idx] += io_time
                # store cache, offload to cpu
                io_costs[cur_idx] += self.profiler.get_dtoh_cost(
                    self.get_compute_cache_size(cur_idx, cache_percent)
                )
        return io_costs

    def get_cost_from_policy(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        costs = np.zeros(self.n + 1)
        io_costs = self.get_io_cost_from_policy(
            cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent
        )

        for cur_idx in range(1, self.n + 1):
            cpu_del = cpu_delegation[cur_idx]
            wait_time = 0
            if self.need_weight(cur_idx):
                io_time_finished = io_costs[weight_prefetch[cur_idx]]
                compute_time_finished = costs[cur_idx - 1]
                # io time larger than compute time
                wait_time = max(0, io_time_finished - compute_time_finished)

            compute_time = wait_time

            if not self.need_cache(cur_idx):
                # mlp layer: compute the result, which is stored in GPU
                compute_time += self.profiler.get_compute_mlp_gpu()
            else:
                # attention layer
                if not cpu_del:
                    # gpu compute
                    io_time_finished = io_costs[cache_prefetch[cur_idx]]
                    compute_time_finished = 0 if cur_idx == 0 else costs[cur_idx - 1]
                    wait_time = max(0, io_time_finished - compute_time_finished)
                    compute_time += wait_time + self.profiler.get_compute_cache_gpu()
                else:
                    compute_time += self.profiler.get_compute_cache_cpu()

            if costs[cur_idx] == 0 and cur_idx != 1:
                costs[cur_idx] = costs[cur_idx - 1]
            costs[cur_idx] += compute_time

        return np.sum(costs)


class DynagenOptOverlappingHeuristic(DynagenOptWorksetHeuristic):
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
        cost_tolerance=1.0,
    ):
        super().__init__(
            num_layers,
            batch_size,
            num_gpu_batches,
            prompt_len,
            gen_len,
            gpu_memory_capacity,
            profiler,
            max_num_prefetch_batches,
        )
        self.cost_tolerance = cost_tolerance

    def get_compute_cache_size(self, c, cache_gpu_percent):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        if j == 0 or j == self.num_layers - 1 or j % 2 == 0:
            return 0
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        return self.profiler.get_cache_size(self.batch_size, self.prompt_len + i) * (100 - cache_gpu_percent) // 100

    def is_cache_offload_valid(self, c, cpu_del):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        return j % 2 == 1 and j != self.num_layers - 1 and cpu_del[c] == 0

    def get_compute_cost(self, c, cpu_del):
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        if i == 0:
            return self.profiler.prefill_batch
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        if j == 0 or j == self.num_layers - 1:
            return 0
        if j % 2 == 0:
            return self.profiler.compute_mlp_gpu
        elif not cpu_del[c]:
            return self.profiler.compute_cache_gpu
        else:
            return self.profiler.compute_cache_cpu
    
    def get_weight_cost(self, c, weight_percent):
        i = (c - 1) // (self.num_layers * self.num_gpu_batches)
        if i == 0:
            return 0
        return self.profiler.get_htod_cost(self.get_compute_weight_size(c, weight_percent))
    
    def get_cache_cost(self, c, cache_percent):
        j = ((c - 1) % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        if j == 0 or j == self.num_layers - 1:
            return 0
        return self.profiler.get_htod_cost(self.get_compute_cache_size(c, cache_percent))

    def optimize_policy(self, position=0, weight_percent=0, cache_percent=0):
        assert type(weight_percent) is int and 0 <= weight_percent <= 100
        assert type(cache_percent) is int and 0 <= cache_percent <= 100

        def get_first_step_remaining(r, c, is_weight):
            min_io_cost_key, min_io_cost = None, float("inf")
            iterator = remaining_sizes.iterkeys(min=r, max=self.gpu_memory_capacity)
            for key in iterator:
                p, _, io_cost = remaining_sizes[key]
                if is_weight and c - p < self.num_layers * self.num_gpu_batches - 1:
                    if io_cost < min_io_cost:
                        min_io_cost = io_cost
                        min_io_cost_key = key
                if not is_weight and c - p < self.num_layers * self.num_gpu_batches:
                    if io_cost < min_io_cost:
                        min_io_cost = io_cost
                        min_io_cost_key = key
            if min_io_cost_key is None:
                return ValueError("No step is found with sufficient remaining VRAM")
            return remaining_sizes[min_io_cost_key]


        weight_prefetch = np.zeros(self.n + 1, np.int64)
        cache_prefetch = np.zeros(self.n + 1, np.int64)
        cpu_del = np.zeros(self.n + 1, np.int32)

        # FIXME: cache_percent always = 100
        mem_consumption = np.sum(self.weight_sizes[1:], dtype=np.uint64) * weight_percent // 100 + self.weight_sizes[0]
        mem_consumption += self.profiler.get_cache_size(self.num_gpu_batches * self.batch_size, self.prompt_len + self.gen_len) * self.profiler.num_hidden_layers * cache_percent // 100
        mem_consumption = int(mem_consumption)
        assert mem_consumption <= self.gpu_memory_capacity, (
            "memory consumption should be less than gpu memory capacity"
        )
        remaining_sizes = QOBTree()

        weight_prefetched = np.zeros(self.n + 1, bool)
        weight_prefetched[: self.num_gpu_batches + 1] = True
        cache_prefetched = np.zeros(self.n + 1, bool)
        for i in range(1, self.n + 1):
            if not self.need_cache(i):
                cache_prefetched[i] = True

        i, p, prev = 1, 1, 1
        with tqdm(total=self.n, position=position, desc="Optimizing", leave=False) as progress:
            while i < self.n:
                io_cost = 0
                compute_cost = self.get_compute_cost(i, cpu_del)
                prefetch_range = self.prefetch_range(p)
                for p in prefetch_range:
                    weight_size = self.get_compute_weight_size(p, weight_percent)
                    cost = self.get_weight_cost(p, weight_percent)
                    if not weight_prefetched[p]:
                        if (
                            mem_consumption + weight_size > self.gpu_memory_capacity
                            or io_cost + cost > compute_cost * (1 + self.cost_tolerance)
                        ):
                            break
                        mem_consumption += weight_size
                        io_cost += cost
                        weight_prefetched[p : p + self.num_gpu_batches] = True
                        weight_prefetch[p] = i
                    if (p - 1) // (self.num_layers * self.num_gpu_batches) == 0:
                        cost = self.get_cache_cost(p, 0)
                    else:
                        cost = self.get_cache_cost(p, cache_percent)
                    cache_size = self.get_compute_cache_size(p, cache_percent)
                    if not cache_prefetched[p]:
                        if (
                            mem_consumption + cache_size > self.gpu_memory_capacity
                            or io_cost + cost > compute_cost * (1 + self.cost_tolerance)
                        ):
                            break
                        mem_consumption += cache_size
                        io_cost += cost
                        cache_prefetched[p] = True
                        cache_prefetch[p] = i
                remaining_sizes.clear()
                for c in range(i, prefetch_range.stop):
                    if i != c:
                        if self.is_weight_offload_valid(c) and weight_prefetched[c]:
                            mem_consumption -= self.get_compute_weight_size(c - 1, weight_percent)
                            assert mem_consumption >= 0
                            r = self.get_remaining_size(mem_consumption)
                            if r not in remaining_sizes:
                                remaining_sizes[r] = (c, mem_consumption, io_cost)
                        if self.is_cache_offload_valid(c, cpu_del) and cache_prefetched[c]:
                            if self._decode(i - 1)[0] == self.gen_len - 1:
                                mem_consumption -= self.get_compute_cache_size(i - 1, 0)
                            else:
                                mem_consumption -= self.get_compute_cache_size(i - 1, cache_percent)
                            assert mem_consumption >= 0
                            r = self.get_remaining_size(mem_consumption)
                            if r not in remaining_sizes:
                                remaining_sizes[r] = (c, mem_consumption, io_cost)
                    if not weight_prefetched[c]:
                        assert i != c, (
                            "weight for the current computing step should not be prefetched in the same step"
                        )
                        weight_size = self.get_compute_weight_size(c, weight_percent)
                        i, mem_consumption, io_cost = get_first_step_remaining(weight_size, c, True)
                        break
                    if not cache_prefetched[c]:
                        if i == c:
                            cpu_del[c] = 1
                            cache_prefetched[c] = True
                        else:
                            cache_size = self.get_compute_cache_size(c, cache_percent)
                            i, mem_consumption, io_cost = get_first_step_remaining(cache_size, c, False)
                        break
                if c == prefetch_range.stop - 1:
                    i = c
                    mem_consumption = self.get_compute_weight_size(i, weight_percent) + (
                        self.get_compute_cache_size(i, cache_percent) if self.need_cache(i) else 0
                    )
                    assert mem_consumption <= self.gpu_memory_capacity
                progress.update(i - prev)
                prev = i
            progress.update(1)


        # for i in tqdm(range(1, self.n + 1), position=position, desc="Optimizing", leave=False):
        #     io_cost = 0
        #     compute_cost = self.get_compute_cost(i, cpu_del)
        #     prefetch_range = self.prefetch_range(p)
        #     for p in prefetch_range:
        #         weight_size = self.get_compute_weight_size(p, weight_percent)
        #         cost = self.get_weight_cost(p, weight_percent)
        #         if not weight_prefetched[p]:
        #             if (
        #                 mem_consumption + weight_size > self.gpu_memory_capacity
        #                 or io_cost + cost > compute_cost * (1 + self.cost_tolerance)
        #             ):
        #                 break
        #             mem_consumption += weight_size
        #             io_cost += cost
        #             weight_prefetched[p : p + self.num_gpu_batches] = True
        #             weight_prefetch[p] = i
        #         if (p - 1) // (self.num_layers * self.num_gpu_batches) == 0:
        #             cache_size = self.get_compute_cache_size(p, 0)
        #         else:
        #             cache_size = self.get_compute_cache_size(p, cache_percent)
        #         cost = self.get_cache_cost(p, cache_percent)
        #         if not cache_prefetched[p]:
        #             if (
        #                 mem_consumption + cache_size > self.gpu_memory_capacity
        #                 or io_cost + cost > compute_cost * (1 + self.cost_tolerance)
        #             ):
        #                 break
        #             mem_consumption += cache_size
        #             io_cost += cost
        #             cache_prefetched[p] = True
        #             cache_prefetch[p] = i
        #     assert weight_prefetched[i], (
        #         "weight for the current computing step should not be prefetched in the same step"
        #     )
        #     if self.is_weight_offload_valid(i):
        #         mem_consumption -= self.get_compute_weight_size(i - 1, weight_percent)
        #     if not cache_prefetched[i]:
        #         cpu_del[i] = 1
        #         cache_prefetched[i] = True
        #     if self.is_cache_offload_valid(i, cpu_del):
        #         if self._decode(i - 1)[0] == self.gen_len - 1:
        #             mem_consumption -= self.get_compute_cache_size(i - 1, 0)
        #         else:
        #             mem_consumption -= self.get_compute_cache_size(i - 1, cache_percent)

        cost = self.get_cost_from_policy(cache_prefetch, weight_prefetch, cpu_del, weight_percent, cache_percent)
        return cost, cache_prefetch, weight_prefetch, cpu_del
