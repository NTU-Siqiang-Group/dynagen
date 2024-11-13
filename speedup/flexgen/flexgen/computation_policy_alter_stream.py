from flexgen.computation_policy_interface import *
from flexgen.timer import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time

def wait_stream_finish(f, need_sync=False):
  f.result()

class CacheLoaderManager:
  def __init__(self, size):
    self.size = size
    self.streams = [torch.cuda.Stream() for _ in range(size)]
    self.executors = ThreadPoolExecutor(max_workers=size)
    self.execute_idx = 0
    
  
  def load_cache(self, func, *args):
    use_stream = self.streams[self.execute_idx]
    self.execute_idx = (self.execute_idx + 1) % self.size
    def _run_func():
      with torch.cuda.stream(use_stream):
        func(*args)
      
      use_stream.synchronize()
  
    return self.executors.submit(_run_func)

class ComputationStreamAlterManager:
  def __init__(self, size):
    self.size = size
    self.streams = [torch.cuda.Stream() for _ in range(size)]
    self.executors = ThreadPoolExecutor(max_workers=size)
    
    def set_stream(use_stream):
      torch.cuda.set_stream(use_stream)
    
    for i in range(size):
      self.executors.submit(set_stream, self.streams[i])
  
  def compute(self, func, *args):
    def _run_func():
      func(*args)
      return torch.cuda.current_stream()

    return self.executors.submit(_run_func)

class ComputationPolicyAlterStream(ComputationPolicyInterface):
  def generation_loop_normal(self, this, evaluate):
    raise NotImplementedError()
  
  def generation_loop_overlap_single_batch(self, this, evaluate, profile_dir):
    def load_layer_weight(i, j, load_to_cpu=False):
      this.load_weight(i, j, 0, overlap=False)
    
    def load_layer_cache(i, j, k, load_to_cpu=False):
      this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)
      
    def compute_layer(i, j, weight_handle, cache_handle):
      if this.layers[j].need_cache:
        wait_stream_finish(cache_handle, True)
        
      cpu_del = this.cpu_del[j]
      this.load_hidden(i, j, 0)
      this.compute_layer(i, j, 0, cpu_delegation=cpu_del)
      this.store_cache(i, j - 1, 0, overlap=False)
      this.store_hidden(i, j, 0)
      this.pop_weight(i, j, 0)
    
    for i in tqdm(range(this.execute_gen_len)):
      timers("generate").start()                    
      this.update_attention_mask(i, 0)
      layers_weights_sync = [None for _ in range(this.num_layers)]
      layers_cache_sync = [None for _ in range(this.num_layers)]
      f = this.cache_loader.load_cache(load_layer_weight, i, 0, this.cpu_del[0])
      layers_weights_sync[0] = f
      for j in range(this.num_layers):
        layers_weights_sync[j].result()
        if this.layers[j].need_cache and this.cpu_del[j]:
          # attention layer and CPU delegation
          # find the next cpu delegation layer
          next_cpu_del = None
          for k in range(j + 1, this.num_layers):
            if this.cpu_del[k]:
              next_cpu_del = k
              break
          end = next_cpu_del if not next_cpu_del is None else this.num_layers
          # load weight and cache
          for k in range(j + 1, end):
            f = this.cache_loader.load_cache(load_layer_weight, i, k, this.cpu_del[k])
            layers_weights_sync[k] = f
            if this.layers[k].need_cache:
              f = this.cache_loader.load_cache(load_layer_cache, i, k, 0, this.cpu_del[k])
              layers_cache_sync[k] = f
        else:
          # GPU attention, MLP, input, or output layer
          # computation in the current thread
          # next layer's weight & cache
          if j + 1 < this.num_layers and layers_weights_sync[j + 1] is None:
            f = this.cache_loader.load_cache(load_layer_weight, i, j + 1, this.cpu_del[j + 1])
            layers_weights_sync[j + 1] = f
            if this.layers[j + 1].need_cache:
              f = this.cache_loader.load_cache(load_layer_cache, i, j + 1, 0, this.cpu_del[j + 1])
              layers_cache_sync[j + 1] = f

        compute_layer(i, j, layers_weights_sync[j], layers_cache_sync[j])
        torch.cuda.current_stream().synchronize()

      timers("generate").stop()

  
  def generation_loop_overlap_multi_batch(self, this, profile_dir):
    def load_layer_weight(i, j):
      this.load_weight(i, j, 0, overlap=False)
      
    def load_layer_cache(i, j, k, load_to_cpu=False):
      this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)
    
    def load_hidden(i, j, k):
      this.load_hidden(i, j, k)
      
    def compute_layer(i, j, k, weight_handle, cache_handle, hidden_handle):
      wait_stream_finish(weight_handle)
      wait_stream_finish(hidden_handle)
      if this.layers[j].need_cache:
        wait_stream_finish(cache_handle)
        
      cpu_del = this.cpu_del[j]
      this.compute_layer(i, j, k, cpu_delegation=cpu_del)
      this.store_cache(i, j, k, overlap=False)
      this.store_hidden(i, j, k)

    for i in tqdm(range(this.execute_gen_len)):
      timers("generate").start()
      
      for k in range(this.num_gpu_batches):
        this.update_attention_mask(i, k)
      
      layers_weights_sync = [None for _ in range(this.num_layers)]
      layer_cache_sync = [[None for _ in range(this.num_gpu_batches)] for _ in range(this.num_layers)]
      layer_hidden_sync = [[None for _ in range(this.num_gpu_batches)] for _ in range(this.num_layers)]
      # load the first weight
      w = this.cache_loader.load_cache(load_layer_weight, i, 0)
      layers_weights_sync[0] = w
      h = this.cache_loader.load_cache(load_hidden, i, 0, 0)
      layer_hidden_sync[0][0] = h
      for j in range(this.num_layers):
        futures = []
        for k in range(this.num_gpu_batches):
          if k != this.num_gpu_batches - 1:
            f = this.stream_manager.compute(compute_layer, i, j, k, layers_weights_sync[j], layer_cache_sync[j][k], layer_hidden_sync[j][k])
            futures.append(f)
          else:
            compute_layer(i, j, k, layers_weights_sync[j], layer_cache_sync[j][k], layer_hidden_sync[j][k])
          
          target_layer = j
          target_batch = k + 1
          if k == this.num_gpu_batches - 1:
            target_layer = j + 1
            target_batch = 0
          
          h = this.cache_loader.load_cache(load_hidden, i, target_layer, target_batch)
          layer_hidden_sync[target_layer][target_batch] = h
        
          if this.layers[target_layer].need_cache:
            # load next batch's cache
            c = this.cache_loader.load_cache(load_layer_cache, i, target_layer, target_batch)
            layer_cache_sync[target_layer][target_batch] = c
          
          if k == 0 and j + 1 < this.num_layers:
            # load next layer's weight
            w = this.cache_loader.load_cache(load_layer_weight, i, j + 1)
            layers_weights_sync[j + 1] = w
        
        # sync
        for f in futures:
          f.result()
        
        this.pop_weight(i, j, 0)
      
      timers("generate").stop()
