import numpy as np


class DynagenOpt:
  def __init__(self, num_layers, batch_size, num_gpu_batches, prompt_len, gen_len):
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.num_gpu_batches = num_gpu_batches
    self.prompt_len = prompt_len
    self.gen_len = gen_len
    
    self.key_matrix_size = 1024 # for one batch, placeholder
    self.value_matrix_size = self.key_matrix_size
    # Assumption 1: the batch can fully saturate the visual memory.
    # 1. Prefetch & offload policy
    #   - KV cache
    #   - Weight
    self.mem_consumption = np.array([None] * gen_len * num_layers * num_gpu_batches) # token * num_batches
    self.cache_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which time the cache is fetched
    self.weight_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches) # at which time the weight is fetched
    # 2. KV cache percentage (initial value, they are fetched into the buffer gradually
    #   according to the prefetch policy)
    self.kv_gpu_percent = np.array([None] * num_layers)
    # 3. CPU delegation
    self.cpu_delegation = np.array([0] * gen_len * num_layers * num_gpu_batches) # which batch should be submitted to CPU

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
            
          cpu_delegation[(i, j, k)] = self.cpu_delegation[self._idx(i, j, k)]
    
    return cache_prefetch, weight_prefetch, cpu_delegation

  def optimize(self):
    pass
  
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

