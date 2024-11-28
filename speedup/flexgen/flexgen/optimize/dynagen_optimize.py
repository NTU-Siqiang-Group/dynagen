import numpy as np
import copy

from flexgen.optimize.network_config import ProfilerConfig

class DynagenOpt:
    def __init__(self, num_layers, batch_size, num_gpu_batches, prompt_len, gen_len, profiler=ProfilerConfig()):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.prompt_len = prompt_len
        self.gen_len = gen_len

        self.profiler = profiler # for one layer, placeholder
        # self.weights_size = self.profiler.get_weights()

        # Assumption 2: the batch can fully saturate the visual memory.
        # 1. Prefetch & offload policy
        #   - KV cache
        #   - Weight
        self.mem_consumption = np.array([None] * gen_len * num_layers * num_gpu_batches) # token * num_batches
        self.cache_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which time the cache is fetched
        # only (i, j, 0) is valid for weight prefetch
        self.weight_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which time the weight is fetched
        # 2. TODO: KV cache percentage and weight percentage (initial value, they are fetched into the buffer gradually
        #   according to the prefetch policy). Currently assume both of them are stored in CPU memory.
        # 3. CPU delegation
        # only (i, j, 0) is valid for cpu delegation
        self.cpu_delegation = np.array([0] * gen_len * num_layers * num_gpu_batches) # which batch should be submitted to 
        # TODO: considering compute two batch at a time. One in GPU and one in CPU

        # GA hyperparameters
        self.mutate_rate = 0.05
        self.cross_rate = 0.5
        self.gen_child_rate = 0.8
        self.gen_mutate_rate = 0.1
        self.population_size = 100
    
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
            population.append((self.cache_prefetch.copy(), self.weight_prefetch.copy(), self.cpu_delegation.copy()))
        population = np.array(population)

        for i in range(max_iter):
            # evaluate the costs
            costs = [self.get_cost_from_policy(*population[i]) for i in range(len(population))]
            # print(f'iter: {i}, best cost: {np.min(costs)}, peak mem consumption: {self.get_mem_consumption(*self.best(population, costs)) / (1<<20)} MB')
            # maintain the best
            best_policy = copy.deepcopy(self.best(population, costs))
            # select
            population = self.select(population, costs, self.population_size)
            # crossover
            pop_num = len(population)
            pop1 = population[:pop_num // 2]
            pop2 = population[pop_num // 2:]
            new_pops = []
            for p1, p2 in zip(pop1, pop2):
                if np.random.rand() <= self.gen_child_rate:
                    new_pops.extend(self.cossover(p1, p2))
                new_pops.append(p1)
                new_pops.append(p2)
            # mutate
            for p in new_pops:
                if np.random.rand() <= self.gen_mutate_rate:
                    self.mutate(p)
            population = np.array(new_pops)
            # replace a random one
            random_pos = np.random.randint(0, len(population))
            population[random_pos] = best_policy
        
        best_policy = self.best(population, costs)
        self.cache_prefetch, self.weight_prefetch, self.cpu_delegation = best_policy

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
    
    def get_random_weight_prefetch_idx(self, i, j):
        # the weight should be loaded for the first batch of each layer
        # the weight does not depend on the token
        left_bound_idx = 0
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
    
    def get_mem_consumption_full(self, cache_prefetch, weight_prefetch, cpu_delegation):
        mem_consumption = np.zeros(self.gen_len * self.num_layers * self.num_gpu_batches)
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                cpu_del = cpu_delegation[self._idx(i, j, 0)]
                for k in range(self.num_gpu_batches):
                    cur_idx = self._idx(i, j, k)
                    if cache_prefetch[cur_idx] is not None and cpu_del:
                        prefetch_idx = cache_prefetch[cur_idx]
                        cache_offload_idx = min(self._idx(i, j, k + 1), len(mem_consumption) - 1)
                        mem_consumption[prefetch_idx:cache_offload_idx] += self.profiler.get_cache_size(self.batch_size, self.prompt_len + i)
                    if weight_prefetch[cur_idx] is not None:
                        weight_prefetch_idx = weight_prefetch[cur_idx]
                        weight_offload_idx = min(self._idx(i, j + 1, 0), len(mem_consumption) - 1)
                        mem_consumption[weight_prefetch_idx:weight_offload_idx] += self.weights_size[j]
        return mem_consumption
        
    def get_mem_consumption(self, cache_prefetch, weight_prefetch, cpu_delegation):
        return np.max(self.get_mem_consumption_full(cache_prefetch, weight_prefetch, cpu_delegation))
    
    def get_cost_from_policy(self, cache_prefetch, weight_prefetch, cpu_delegation):
        costs = np.zeros(self.gen_len * self.num_layers * self.num_gpu_batches)
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                is_weight_loaded = False
                need_cache = self.need_cache(j)
                cpu_del = cpu_delegation[self._idx(i, j, 0)]
                for k in range(self.num_gpu_batches):
                    cur_idx = self._idx(i, j, k)
                    if not is_weight_loaded and (i != 0 and j != 0):
                        # probably need to wait weight
                        weight_prefetch_idx = weight_prefetch[self._idx(i, j, 0)]
                        sum_cost = np.sum(costs[weight_prefetch_idx:cur_idx])
                        wait_time = max(0, self.get_htod_cost(self.weights_size[j]) - sum_cost)
                        is_weight_loaded = True
                        costs[cur_idx] += wait_time
                    
                    if not need_cache:
                        # mlp layer
                        costs[cur_idx] += self.get_compute_mlp_gpu() + self.get_dtoh_cost(self.profiler.get_hidden_size(self.batch_size, self.prompt_len + i))
                    else:
                        # attention layer
                        if not cpu_del:
                            # gpu compute
                            # probably need to wait
                            cache_prefetch_idx = cache_prefetch[cur_idx]
                            sum_cost = np.sum(costs[cache_prefetch_idx:cur_idx])
                            wait_time = max(0, self.get_htod_cost(self.profiler.get_cache_size(self.batch_size, self.prompt_len + i)) - sum_cost)
                            costs[cur_idx] += wait_time + self.get_compute_cache_gpu()
                            # computation cost
                            costs[cur_idx] += self.get_compute_cache_gpu() + self.get_dtoh_cost(self.profiler.get_hidden_size(self.batch_size, self.prompt_len + i))
                        else:
                            # cpu compute
                            costs[cur_idx] += self.get_compute_cache_cpu()
        return np.sum(costs)
    
    def cossover(self, p1, p2):
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

        return (p1_cache_prefetch, p1_weight_prefetch, p1_cpu_delegation), (p2_cache_prefetch, p2_weight_prefetch, p2_cpu_delegation)

    
    def select(self, population, costs, population_size):
        # normalize costs
        prob = (np.max(costs) - costs + 1) / np.mean(costs)
        # normalize to 0-1
        prob = prob / np.sum(prob)
        # randomly evict some policies
        idxs = np.random.choice(population.shape[0], size=population_size, p=prob)
        return population[idxs]

    def mutate(self, policy):
        # shallow copy
        cache_prefetch, weight_prefetch, cpu_delegation = policy
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
    
    def best(self, population, costs):
        return population[np.argmin(costs)]

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
                        if layers_weights_sync[batch][layer] is None and loading_weights < self.num_gpu_batches * 4:
                            self.weight_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            layers_weights_sync[batch][layer] = 1
                            loading_weights += 1
                        if layers_cache_sync[batch][layer] is None and loading_caches < 4:
                            self.cache_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            self.cpu_delegation[self._idx(token, layer, batch)] = batch % 2 == 0
                            layers_cache_sync[batch][layer] = 1
                            loading_caches += 1
                    # compute
                    layers_weights_sync[k][j] = None
                    layers_cache_sync[k][j] = None

