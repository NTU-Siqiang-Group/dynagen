from math import ceil
from tqdm import tqdm
import numpy as np
import copy

from flexgen.optimize.network_config import ProfilerConfig

from concurrent.futures import ThreadPoolExecutor

class CompThreads:
    def __init__(self, num_threads=32):
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
    
    def run(self, func, *args):
        return self.pool.submit(func, *args)

class DynagenOpt:
    def __init__(self, num_layers, batch_size, num_gpu_batches, prompt_len, gen_len, profiler=ProfilerConfig()):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.prompt_len = prompt_len
        self.gen_len = gen_len

        self.profiler = profiler # for one layer, placeholder
        self.weights_size = self.profiler.get_weights()

        # Assumption 2: the batch can fully saturate the GPU memory.
        # 1. Prefetch & offload policy
        #   - KV cache
        #   - Weight
        self.mem_consumption = np.array([None] * gen_len * num_layers * num_gpu_batches) # token * num_batches
        self.cache_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which "step" the cache is fetched
        # only (i, j, 0) is valid for weight prefetch
        self.weight_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which "step" the weight is fetched
        # 2. TODO: KV cache percentage and weight percentage (initial value, they are fetched into the buffer gradually
        #   according to the prefetch policy). Currently assume both of them are stored in CPU memory.
        # 3. CPU delegation
        # only (i, j, 0) is valid for cpu delegation
        self.cpu_delegation = np.array([0] * gen_len * num_layers * num_gpu_batches) # which batch should be submitted to 
        # TODO: considering compute two batch at a time. One in GPU and one in CPU

        self.weight_percent = np.zeros(num_layers)
        self.cache_percent = np.zeros(num_layers)
        # GA hyperparameters
        self.mutate_rate = 0.05
        self.cross_rate = 0.5
        self.gen_child_rate = 0.8
        self.gen_mutate_rate = 0.1
        self.population_size = 100
        
        self.comp_threads = CompThreads()
    
    def get_htod_cost(self, size):
        return self.profiler.get_htod_cost(size)
    
    def get_dtoh_cost(self, size):
        return self.profiler.get_dtoh_cost(size)
    
    def get_compute_cache_gpu(self):
        return self.profiler.get_compute_cache_gpu()
    
    def get_compute_cache_cpu(self):
        return self.profiler.get_compute_cache_cpu()
    
    def get_compute_mlp_gpu(self):
        return self.profiler.get_compute_mlp_gpu()

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
    
    # TODO: considering the KV cache is stored in GPU
    # TODO: considering cpu delegation for batch
    def optimize(self, max_iter=10):
        population = []
        # init population
        for _ in range(self.population_size):
            self.init_cache_prefetch()
            self.init_weight_prefetch()
            self.init_cpu_delegation()
            self.init_weight_percent()
            self.init_cache_percent()
            population.append((self.cache_prefetch.copy(), self.weight_prefetch.copy(), self.cpu_delegation.copy(), self.weight_percent.copy(), self.cache_percent.copy()))


        for i in tqdm(range(max_iter)):
            # evaluate the costs
            costs = [0 for _ in range(len(population))]
            mems = [0 for _ in range(len(population))]
            futures = []
            for j in range(len(population)):
                def get_cost(k):
                    costs[k] = self.get_cost_from_policy(*population[k])
                    # mems[k] = self.get_mem_consumption(*population[k])
                f = self.comp_threads.run(get_cost, j)
                futures.append(f)
            # wait for all the threads
            for f in futures:
                f.result()
            
            # maintain the best
            best_idx = self.best(population, costs)
            print(f'iter {i}, best cost: {costs[best_idx]}, peak mem: {self.get_mem_consumption(*population[best_idx]) / (1 << 20)} MB')
            best_policy = copy.deepcopy(population[best_idx])
            # select
            population = self.select(population, costs, self.population_size)
            # crossover
            pop_num = len(population)
            pop1 = population[:pop_num // 2]
            pop2 = population[pop_num // 2:]
            new_pops = []
            futures = []
            for p1, p2 in zip(pop1, pop2):
                if np.random.rand() <= self.gen_child_rate:
                    def crossover_compute():
                        return self.crossover(p1, p2)
                    # new_pops.extend(self.cossover(p1, p2))
                    f = self.comp_threads.run(crossover_compute)
                    futures.append(f)
                new_pops.append(p1)
                new_pops.append(p2)
            for f in futures:
                new_pops.extend(f.result())
            # mutate
            for p in new_pops:
                if np.random.rand() <= self.gen_mutate_rate:
                    self.mutate(p)
            population = new_pops
            # replace a random one
            random_pos = np.random.randint(0, len(population))
            population[random_pos] = best_policy
        
        best_policy = population[self.best(population, costs)]
        self.cache_prefetch, self.weight_prefetch, self.cpu_delegation, self.weight_percent, self.cache_percent = best_policy

    def get_random_cache_prefetch_idx(self, i, j, k):
        left_bound = (i - 1, j, k)
        if i == 0:
            left_bound = None
        right_bound = (i, j, k)
        right_bound_idx = self._idx(*right_bound)
        left_bound_idx = 0 if left_bound is None else self._idx(*left_bound)
        return np.random.randint(left_bound_idx, right_bound_idx)

    def init_cache_prefetch(self):
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                if not self.need_cache(j):
                    continue
                for k in range(self.num_gpu_batches):
                    self.cache_prefetch[self._idx(i, j, k)] = self.get_random_cache_prefetch_idx(i, j, k)
    
    def init_weight_percent(self):
        for j in range(self.num_layers):
            self.weight_percent[j] = np.random.rand() # 0 ~ 1
    
    def init_cache_percent(self):
        for j in range(self.num_layers):
            self.cache_percent[j] = np.random.rand()
    
    def get_random_weight_prefetch_idx(self, i, j):
        # the weight should be loaded for the first batch of each layer
        # the weight does not depend on the token
        left_bound_idx = 0
        if i > 0:
            left_bound_idx = self._idx(i - 1, j, 0)
        right_bound_idx = self._idx(i, j, 0)
        if right_bound_idx == 0:
            return 0
        # randomly choose a time to load the weight
        return np.random.randint(left_bound_idx, right_bound_idx)

    def init_weight_prefetch(self):
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                self.weight_prefetch[self._idx(i, j, 0)] = self.get_random_weight_prefetch_idx(i, j)
    
    # TODO: considering cpu delegation for batch rather than layers?
    def get_random_cpu_delegation(self, j):
        if not self.need_cache(j):
            return 0
        return np.random.randint(0, 2)
    
    def init_cpu_delegation(self):
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                if not self.need_cache(j):
                    continue
                use_del = self.get_random_cpu_delegation(j)
                self.cpu_delegation[self._idx(i, j, 0)] = use_del
    
    def get_mem_consumption_full(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        mem_consumption = np.zeros(self.gen_len * self.num_layers * self.num_gpu_batches)
        for j in range(self.num_layers):
            mem_consumption += self.weights_size[j] * weight_percent[j]
            if self.need_cache(j):
                mem_consumption += self.profiler.get_cache_size(self.batch_size, self.prompt_len + self.gen_len) * cache_percent[j]
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                cpu_del = cpu_delegation[self._idx(i, j, 0)]
                for k in range(self.num_gpu_batches):
                    cur_idx = self._idx(i, j, k)
                    if cache_prefetch[cur_idx] is not None and cpu_del:
                        prefetch_idx = cache_prefetch[cur_idx]
                        cache_offload_idx = min(self._idx(i, j, k + 1), len(mem_consumption) - 1)
                        mem_consumption[prefetch_idx:cache_offload_idx] += self.profiler.get_cache_size(self.batch_size, self.prompt_len + i) * (1 - cache_percent[j])
                    if weight_prefetch[cur_idx] is not None:
                        weight_prefetch_idx = weight_prefetch[cur_idx]
                        weight_offload_idx = min(self._idx(i, j + 1, 0), len(mem_consumption) - 1)
                        mem_consumption[weight_prefetch_idx:weight_offload_idx] += self.weights_size[j] * (1 - weight_percent[j])
        return mem_consumption
        
    def get_mem_consumption(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        return np.max(self.get_mem_consumption_full(cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent))
    
    def get_io_cost_from_policy(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        io_costs = np.zeros(self.gen_len * self.num_layers * self.num_gpu_batches)
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                need_cache = self.need_cache(j)
                cpu_del = cpu_delegation[self._idx(i, j, 0)]
                for k in range(self.num_gpu_batches):
                    cur_idx = self._idx(i, j, k)
                    if k == 0 and (i != 0 and j != 0):
                        # wait for the weight loaded
                        weight_prefetch_idx = weight_prefetch[self._idx(i, j, 0)]
                        if io_costs[weight_prefetch_idx] == 0:
                            io_costs[weight_prefetch_idx] = 0 if weight_prefetch_idx == 0 else io_costs[weight_prefetch_idx - 1]
                        _, prefetch_j, _ = self._decode(weight_prefetch_idx)
                        if prefetch_j != j:
                            io_time = self.get_htod_cost(self.weights_size[j] * (1 - weight_percent[j]))
                            io_costs[weight_prefetch_idx] += io_time
                    # store cache
                    if not need_cache:
                        # mlp layer does not request IO
                        continue
                    if io_costs[cur_idx] == 0 and cur_idx != 0:
                        io_costs[cur_idx] = io_costs[cur_idx - 1]
                    if cpu_del:
                        # 1. TODO: the cache is current stored in CPU, no htod cost is required
                        # 2. the new k v should be transfered back to CPU (dtoh)
                        io_costs[cur_idx] += self.get_dtoh_cost(self.profiler.get_cache_size(self.batch_size, 1))
                    else:
                        # 1. TODO: the cache is stored in CPU, htod cost is required
                        # 2. the new k v is not required to transfer to CPU. But the whole KV cache should be transfered to CPU after computation
                        cache_prefetch_idx = cache_prefetch[cur_idx]
                        if io_costs[cache_prefetch_idx] == 0:
                            io_costs[cache_prefetch_idx] = 0 if cache_prefetch_idx == 0 else io_costs[cache_prefetch_idx - 1]
                        io_time = self.get_htod_cost(self.profiler.get_cache_size(self.batch_size, self.prompt_len + i) * (1 - cache_percent[j]))
                        # prefetch cache
                        io_costs[cache_prefetch_idx] += io_time
                        # store cache, offload to cpu
                        io_costs[cur_idx] += self.get_dtoh_cost(self.profiler.get_cache_size(self.batch_size, self.prompt_len + i) * (1 - cache_percent[j]))
        return io_costs
    
    def get_cost_from_policy(self, cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent):
        costs = np.zeros(self.gen_len * self.num_layers * self.num_gpu_batches)
        io_costs = self.get_io_cost_from_policy(cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent)
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                need_cache = self.need_cache(j)
                cpu_del = cpu_delegation[self._idx(i, j, 0)]
                for k in range(self.num_gpu_batches):
                    cur_idx = self._idx(i, j, k)
                    wait_time = 0
                    if k == 0 and (i != 0 and j != 0):
                        io_time_finished = io_costs[weight_prefetch[cur_idx]]
                        compute_time_finished = costs[cur_idx - 1]
                        # io time larger than compute time
                        wait_time = max(0, io_time_finished - compute_time_finished)
                    
                    compute_time = wait_time

                    if not need_cache:
                        # mlp layer: compute the result, which is stored in GPU
                        compute_time += self.get_compute_mlp_gpu()
                    else:
                        # attention layer
                        if not cpu_del:
                            # gpu compute
                            io_time_finished = io_costs[cache_prefetch[cur_idx]]
                            compute_time_finished = 0 if cur_idx == 0 else costs[cur_idx - 1]
                            wait_time = max(0, io_time_finished - compute_time_finished)
                            compute_time += wait_time + self.get_compute_cache_gpu()
                        else:
                            compute_time += self.get_compute_cache_cpu()
                    
                    if costs[cur_idx] == 0 and cur_idx != 0:
                        costs[cur_idx] = costs[cur_idx - 1]
                    costs[cur_idx] += compute_time
        
        return np.sum(costs)
    
    def crossover(self, p1, p2):
        pos = np.random.rand(self.cache_prefetch.shape[0]) <= self.cross_rate
        p1_cache_prefetch = p1[0].copy()
        p2_cache_prefetch = p2[0].copy()
        p1_weight_prefetch = p1[1].copy()
        p2_weight_prefetch = p2[1].copy()
        p1_cpu_delegation = p1[2].copy()
        p2_cpu_delegation = p2[2].copy()
        # swap
        p1_cache_prefetch[pos], p2_cache_prefetch[pos] = p2_cache_prefetch[pos], p1_cache_prefetch[pos]
        p1_weight_prefetch[pos], p2_weight_prefetch[pos] = p2_weight_prefetch[pos], p1_weight_prefetch[pos]
        p1_cpu_delegation[pos], p2_cpu_delegation[pos] = p2_cpu_delegation[pos], p1_cpu_delegation[pos]
        
        pos = np.random.rand(self.weight_percent.shape[0]) <= self.cross_rate
        p1_weight_percent = p1[3].copy()
        p2_weight_percent = p2[3].copy()
        p1_cache_percent = p1[4].copy()
        p2_cache_percent = p2[4].copy()
        p1_weight_percent[pos], p2_weight_percent[pos] = p2_weight_percent[pos], p1_weight_percent[pos]
        p1_cache_percent[pos], p2_cache_percent[pos] = p2_cache_percent[pos], p1_cache_percent[pos]

        return (p1_cache_prefetch, p1_weight_prefetch, p1_cpu_delegation, p1_weight_percent, p1_cache_percent), (p2_cache_prefetch, p2_weight_prefetch, p2_cpu_delegation, p2_weight_percent, p2_cache_percent)

    
    def select(self, population, costs, population_size):
        # normalize costs
        prob = (np.max(costs) - costs + 1) / np.mean(costs)
        # normalize to 0-1
        prob = prob / np.sum(prob)
        # randomly evict some policies
        idxs = np.random.choice(len(population), size=population_size, p=prob)
        population = [population[i] for i in idxs]
        return population

    def mutate(self, policy):
        # shallow copy
        cache_prefetch, weight_prefetch, cpu_delegation, weight_percent, cache_percent = policy
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                use_del = self.get_random_cpu_delegation(j)
                if np.random.rand() <= self.mutate_rate:
                    cpu_delegation[self._idx(i, j, 0)] = use_del
                need_cache = self.need_cache(j)
                for k in range(self.num_gpu_batches):
                    if not np.random.rand() > self.mutate_rate:
                        continue
                    cur_idx = self._idx(i, j, k)
                    if need_cache:
                        cache_prefetch[cur_idx] = self.get_random_cache_prefetch_idx(i, j, k)
                    weight_prefetch[cur_idx] = self.get_random_weight_prefetch_idx(i, j)
        
        for j in range(self.num_layers):
            if np.random.rand() <= self.mutate_rate:
                weight_percent[j] = np.random.rand()
            if np.random.rand() <= self.mutate_rate:
                cache_percent[j] = np.random.rand()
    
    def best(self, population, costs):
        return np.argmin(costs)

    def need_cache(self, layer):
        return layer != self.num_layers - 1 and layer % 2 != 0

    def _idx(self, token, layer, batch):
        return token * self.num_layers * self.num_gpu_batches + layer * self.num_gpu_batches + batch

    def _decode(self, idx):
        token = idx // (self.num_layers * self.num_gpu_batches)
        layer = (idx % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        batch = idx % self.num_gpu_batches
        return (token, layer, batch)

    def optimize_alter_v2(self):
        layers_weights_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_cache_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_weights_sync[0][0] = 1
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    loading_weights = sum(x is not None for sublist in layers_weights_sync for x in sublist)
                    loading_caches = sum(x is not None for sublist in layers_cache_sync for x in sublist)
                    for l in range(k + 1, k + self.num_gpu_batches * 10):
                        batch = l % self.num_gpu_batches
                        layer = j + l // self.num_gpu_batches
                        token = i
                        if layer >= self.num_layers:
                            layer = layer - self.num_layers
                            token = i + 1
                        if token >= self.gen_len:
                            continue
                        if layers_weights_sync[batch][layer] is None and loading_weights < self.num_gpu_batches * 5:
                            self.weight_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            layers_weights_sync[batch][layer] = 1
                            loading_weights += 1
                        if layers_cache_sync[batch][layer] is None and loading_caches < 15:
                            self.cache_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            self.cpu_delegation[self._idx(token, layer, batch)] = batch % 2 == 0
                            layers_cache_sync[batch][layer] = 1
                            loading_caches += 1
                    # compute
                    layers_weights_sync[k][j] = None
                    layers_cache_sync[k][j] = None


# Greedy / DP?:
# 1. 只有奇数层，且不是最后一层需要cache
# 2. 推论：[i, j, k]需要的cache只可以在 >= [i-1, j, k+1] 且 < [i, j, k]的step被prefetch
# 3. 第 [i, j, k] "step" **可以且只可以** prefetch之后任意 **一整层** 的weight，假设在[0, 0, 0]prefetch了第1层的weight，那weight_prefetch[0, 1, 0] = 0。特别地，[0, 0, k]的weight通常已被预加载，无需考虑。
# 4. 推论：[i, j, 0]需要的weight只可以在 >= [i-1, j+1, 0] 且 < [i, j, 0]的step被prefetch
# 5. weight必须被prefetch到GPU；但cache可以不被prefetch到GPU，此时需要使用cpu_delegation
# 6. cache在下一个batch被offload，weight在下一个layer的第0个batch被offload
# 7. 推论：第[0, 0, k] "step" 的latency只取决于：compute<-是否cpu_delegation
# 8. 加入对GPU memory的大小限制
# 9. 假设：在一个"step"里prefetch再多的weight或cache都不会影响这个"step"的latency，只会影响它的mem_consumption
# 10. 假设：weight_percent_gpu = 0, cache_percent_gpu = 0
# 11. 数据结构：
#   "items" / "steps" :=
#       +----------+---------------------------------------+
#       |0         |WWCCCCC...CCWWCCCCC...CCWWCCCCC...CCWW |
#       +----------+---------------------------------------+
#       |1         |WWCCCCC...CCWWCCCCC...CCWWCCCCC...CCWW |
#       +----------+---------------------------------------+
#       |...       |...                                    |
#       +----------+---------------------------------------+
#       |gen_len-1 |WWCCCCC...CCWWCCCCC...CCWWCCCCC...CCWW |
#       +----------+---------------------------------------+
#   gpu_memory_capacity: int    # 显存容量（GiB）
#   cache_count = gen_len * ceil((num_layers - 2) / 2) * num_gpu_batches
#   weight_count = gen_len * num_layers
#   latency = np.full((cache_count + weight_count + 1, gpu_memory_capacity + 1), -np.inf)
#   prefetch = np.empty(n)      # 一个step应该在哪个step被prefetch
#   mem_consumption_cumsum先填入每个"item"的大小（Byte），再计算前缀和，最后转换为GiB
# 12. 初始化：
# latency[0] = 00000...0 -> 不计时的weight prefetch "base step"
# latency[i][0...gpu_memory_lower_bound-1] = 0

# latency[i] = latency[i - 1] + max(weight_loading_finished[i] - latency[i - 1], 0)
#                             + max(cache_loading_finished[i] - latency[i - 1], 0) + compute[i]
# min(latency[-1])


class DynagenOptDP:
    # Counts from 1
    class BinaryIndexedTree:
        def __init__(self, size):
            self.size = size
            self.tree = np.zeros(size + 1)

        def update(self, index, delta):
            assert index > 0
            while index <= self.size:
                self.tree[index] += delta
                index += index & (-index)

        def get_cumsum(self, index):
            result = 0
            while index > 0:
                result += self.tree[index]
                index -= index & (-index)
            return result


    # Counts from 0
    class Steps:
        def __init__(self, steps, num_layers):
            self.steps = steps
            self.num_layers = num_layers

        def __iter__(self):
            start = 0
            for i, end in enumerate(self.steps):
                yield (i, start, end, i % 2 == 0)  # seg_idx, start, end, is_weight
                start = end

        def riter(self, seg_idx, lower_bound):
            while seg_idx >= 0:
                start = self.steps[seg_idx - 1] if seg_idx > 0 else 0
                end = self.steps[seg_idx]
                yield (seg_idx, start, end, seg_idx % 2 == 0)  # seg_idx, start, end, is_weight
                if start <= lower_bound:
                    break
                seg_idx -= 1


    def __init__(self, num_layers, batch_size, num_gpu_batches, prompt_len, gen_len, gpu_memory_capacity, profiler=ProfilerConfig()):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches

        cache_count = gen_len * ceil((num_layers - 2) / 2) * num_gpu_batches
        weight_count = gen_len * num_layers
        self.n = cache_count + weight_count
        self.step_per_token = self.n // gen_len
        self.gpu_memory_capacity = gpu_memory_capacity

        steps = np.empty((num_layers - 1) + (num_layers - 2) * (gen_len - 1), np.uint32)
        start = 0
        for seg_idx in range(len(steps)):
            if seg_idx > 0 and seg_idx < len(steps) - 1 and seg_idx % (num_layers - 2) == 0:
                steps[seg_idx] = start + 4
                start = steps[seg_idx]
                continue
            if seg_idx % 2 == 0:
                steps[seg_idx] = start + 2
                start = steps[seg_idx]
                continue
            steps[seg_idx] = start + num_gpu_batches
            start = steps[seg_idx]
        self.steps = self.Steps(steps, num_layers)

        mem_consumption = np.empty(self.n + 1, np.uint32)
        mem_consumption[0] = 0
        weight_sizes = np.array(profiler.get_weights())
        i, j = 0, 0
        for seg_idx, start, end, is_weight in self.steps:
            if not is_weight:
                mem_consumption[start + 1:end + 1] = profiler.get_cache_size(batch_size, prompt_len + i)
                continue
            for c in range(start, end):
                if j == num_layers - 1:
                    mem_consumption[c+1] = weight_sizes[j]
                    i += 1
                    j = 0
                else:
                    mem_consumption[c+1] = weight_sizes[j]
                    j += 1
        self.mem_consumption_cumsum = np.cumsum(mem_consumption).astype(np.uint64)  # Counts from 1
        weight_sizes = weight_sizes + np.insert(weight_sizes[:-1], 0, 0)
        self.gpu_memory_lower_bound = np.ceil(np.max(weight_sizes) / (1 << 30)).astype(np.int32)

        self.latency = np.full((self.n + 1, gpu_memory_capacity + 1), -np.inf)  # Counts from 1
        self.latency[0] = 0
        self.prefetch = np.full(self.n + 1, -1, np.int32)  # Counts from 1
        self.io_costs = self.BinaryIndexedTree(self.n)  # 某一步的io部分需要的时间组成的线段数组
        self.profiler = profiler

    # Counts from 0
    def _decode(self, c):
        i = c // self.step_per_token

        if c % self.step_per_token >= self.step_per_token - 2:
            seg_idx = (self.num_layers - 2) * (i + 1)
            is_weight = True
        else:
            seg_idx = (self.num_layers - 2) * i + (c % self.step_per_token) // (2 + self.num_gpu_batches) * 2
            is_weight = (c % self.step_per_token) % (2 + self.num_gpu_batches) < 2
            if not is_weight:
                seg_idx += 1
        start = 0 if seg_idx == 0 else self.steps.steps[seg_idx - 1]

        k = 0 if is_weight else c - start

        if c % self.step_per_token >= self.step_per_token - 2:
            seg_idx = self.num_layers - 2
        else:
            seg_idx %= self.num_layers - 2
        j = seg_idx if not is_weight else seg_idx + c - start
        return int(i), int(j), int(k)

    # Counts from 1
    def get_mem_consumption(self, c, is_weight, weight_step=None):
        if not is_weight and weight_step is not None:
            result = self.mem_consumption_cumsum[c] - self.mem_consumption_cumsum[c - 1] + self.mem_consumption_cumsum[weight_step] - self.mem_consumption_cumsum[weight_step - 1]
        else:
            result = self.mem_consumption_cumsum[c] - self.mem_consumption_cumsum[c - 1]
        return int(np.ceil(result / (1 << 30)))

    # Counts from 1
    def get_mem_consumption_segsum(self, i, c, is_i_weight, weight_step=None):
        if not is_i_weight and weight_step is not None:
            result = self.mem_consumption_cumsum[c] - self.mem_consumption_cumsum[i - 1] + self.mem_consumption_cumsum[weight_step] - self.mem_consumption_cumsum[weight_step - 1]
        else:
            result = self.mem_consumption_cumsum[c] - self.mem_consumption_cumsum[i - 1]
        return int(np.ceil(result / (1 << 30)))

    def get_compute(self, is_weight, cpu_del=False):
        if is_weight:
            assert not cpu_del
            return self.profiler.get_compute_mlp_gpu()
        if cpu_del:
            return self.profiler.get_compute_cache_cpu()
        return self.profiler.get_compute_cache_gpu()

    # Counts from 0
    def get_weight_prefetch_lower_bound(self, c, is_seg_end):
        if is_seg_end:
            return max(c - self.step_per_token + self.num_gpu_batches + 1, 0)
        return max(c - self.step_per_token + 1, 0)

    # Counts from 1
    def get_weight_loading_finished(self, prefetch_step, c):
        return self.io_costs.get_cumsum(prefetch_step) + self.profiler.get_htod_cost(self.get_mem_consumption(c, True) << 30)

    # Counts from 0
    def get_cache_prefetch_lower_bound(self, c):
        return max(c - self.step_per_token + 1, 0)

    # Counts from 1
    def get_cache_loading_finished(self, prefetch_step, c):
        return self.io_costs.get_cumsum(prefetch_step) + self.profiler.get_htod_cost(self.get_mem_consumption(c, False) << 30)

    # Counts from 0
    def update_io_costs(self, c, is_weight):
        prefetch_step = self.prefetch[c]
        cpu_del = prefetch_step == -1
        if not cpu_del:
            self.io_costs.update(prefetch_step + 1, self.profiler.get_htod_cost(self.get_mem_consumption(c + 1, is_weight) << 30))
        if not is_weight:
            if cpu_del:
                self.io_costs.update(c + 1, self.profiler.get_dtoh_cost(self.profiler.get_cache_size(self.batch_size, 1)))
            else:
                self.io_costs.update(c + 1, self.profiler.get_dtoh_cost(self.get_mem_consumption(c + 1, False) << 30))

    def optimize(self):
        for seg_idx, start, end, is_weight in self.steps:
            for c in range(start, end):
                for r in range(self.gpu_memory_capacity, max(-1, self.gpu_memory_lower_bound - 2), -1):
                    if r < self.get_mem_consumption(c + 1, is_weight, start):
                        if c == 0:
                            continue
                        if self.prefetch[c] != -1:
                            self.update_io_costs(c, is_weight)
                        else:
                            assert not is_weight  # gpu_memory_lower_bound should guarantee that GPU is at least capable of holding the weight
                            self.latency[c + 1][r] = self.latency[c][r] - self.get_compute(is_weight, True)  # CPU delegation
                    else:
                        if is_weight:
                            _c = max(c - 1, 0)
                            f_seg_idx = seg_idx if _c >= start else seg_idx - 1
                            weight_prefetch_lower_bound = self.get_weight_prefetch_lower_bound(c, c == end - 1)
                            for _, i_start, i_end, is_i_weight in self.steps.riter(f_seg_idx, weight_prefetch_lower_bound):
                                for i in range(min(_c, i_end - 1), max(weight_prefetch_lower_bound - 1, i_start - 1), -1):
                                    mem_consumption_segsum = self.get_mem_consumption_segsum(i + 1, c + 1, is_i_weight, i_start)
                                    if r < mem_consumption_segsum:
                                        self.update_io_costs(c, is_weight)
                                        break
                                    weight_loading_finished = self.get_weight_loading_finished(i + 1, c + 1)
                                    _r = 1 if r - mem_consumption_segsum == 0 else max(0, r - mem_consumption_segsum)
                                    weight_loading_latency = min(weight_loading_finished + self.latency[c][_r], 0)
                                    new_latency = self.latency[c][_r] + weight_loading_latency - self.get_compute(is_weight)
                                    if self.latency[c + 1][r] < new_latency:
                                        self.latency[c + 1][r] = new_latency
                                        if r == self.gpu_memory_capacity:
                                            self.prefetch[c] = i
                            continue

                        weight_step = start - 1
                        mem_consumption = self.get_mem_consumption(weight_step + 1, True)
                        if r < mem_consumption:
                            continue
                        _r = 1 if r - mem_consumption == 0 else max(0, r - mem_consumption)
                        self.latency[c + 1][_r] = self.latency[c][_r] - self.get_compute(is_weight, True)

                        f_seg_idx = seg_idx if c - 1 >= start else seg_idx - 1
                        cache_prefetch_lower_bound = self.get_cache_prefetch_lower_bound(c)
                        for _, i_start, i_end, is_i_weight in self.steps.riter(f_seg_idx, cache_prefetch_lower_bound):
                            for i in range(min(c - 1, i_end - 1), max(cache_prefetch_lower_bound - 1, i_start - 1), -1):
                                mem_consumption_segsum = self.get_mem_consumption_segsum(i + 1, c + 1, is_i_weight, i_start)
                                if r < mem_consumption_segsum:
                                    self.update_io_costs(c, is_weight)
                                    break
                                cache_loading_finished = self.get_cache_loading_finished(i + 1, c + 1)
                                _r = 1 if r - mem_consumption_segsum == 0 else max(0, r - mem_consumption_segsum)
                                cache_loading_latency = min(cache_loading_finished + self.latency[c][_r], 0)
                                new_latency = self.latency[c][_r] + cache_loading_latency - self.get_compute(is_weight, False)
                                if self.latency[c + 1][r] < new_latency:
                                    self.latency[c + 1][r] = new_latency
                                    if r == self.gpu_memory_capacity:
                                        self.prefetch[c] = i

    def get_policy(self):
        cache_prefetch = {}
        weight_prefetch = {}
        cpu_delegation = {}
        i, j = 0, 0
        for _, start, end, is_weight in self.steps:
            if not is_weight:
                k = 0
                for c in range(start, end):
                    prefetch_step = self.prefetch[c]
                    if prefetch_step == -1:
                        cpu_delegation[(i, j - 1, k)] = 1
                    else:
                        cpu_delegation[(i, j - 1, k)] = 0
                        cache_prefetch.setdefault(self._decode(prefetch_step), []).append((i, j - 1, k))
                    k += 1
                continue
            for c in range(start, end):
                prefetch_step = self.prefetch[c]
                assert prefetch_step != -1
                weight_prefetch.setdefault(self._decode(prefetch_step), []).append((i, j, 0))
                if j == self.num_layers - 1:
                    i += 1
                    j = 0
                else:
                    j += 1

        return cache_prefetch, weight_prefetch, cpu_delegation
