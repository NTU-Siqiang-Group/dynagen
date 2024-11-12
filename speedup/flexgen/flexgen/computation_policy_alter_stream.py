from flexgen.computation_policy_interface import *
from flexgen.timer import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time

def wait_stream_finish(f):
  f.result()

class CacheLoaderManager:
  def __init__(self, size):
    self.size = size
    self.executors = ThreadPoolExecutor(max_workers=size)
  
  def load_cache(self, i, func, *args):
    def _run_func():
      func(*args)
    
    return self.executors.submit(_run_func)

class ComputationStreamAlterManager:
  def __init__(self, size):
    self.size = size
    self.executors = ThreadPoolExecutor(max_workers=size)
  
  def compute(self, i, func, *args):
    def _run_func():
      func(*args)

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
      wait_stream_finish(weight_handle)
      if this.layers[j].need_cache:
        wait_stream_finish(cache_handle)
        
      cpu_del = this.cpu_del[j]
      this.load_hidden(i, j, 0)
      this.compute_layer(i, j, 0, cpu_delegation=cpu_del)
      this.store_cache(i, j, 0, overlap=False)
      this.store_hidden(i, j, 0)
      this.pop_weight(i, j, 0)
    timers("generate").start()
    
    for i in tqdm(range(this.execute_gen_len)):
      this.update_attention_mask(i, 0)
      layers_weights_sync = [None for _ in range(this.num_layers)]
      layers_cache_sync = [None for _ in range(this.num_layers)]
      f = this.cache_loader.load_cache(0, load_layer_weight, i, 0, this.cpu_del[0])
      layers_weights_sync[0] = f
      for j in range(this.num_layers):
        compute_f = None
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
            f = this.cache_loader.load_cache(k % this.cache_loader.size, load_layer_weight, i, k, this.cpu_del[k])
            layers_weights_sync[k] = f            
            if this.layers[k].need_cache:
              f = this.cache_loader.load_cache(k % this.cache_loader.size, load_layer_cache, i, k, 0, this.cpu_del[k])
              layers_cache_sync[k] = f
          # compute this layer
          compute_f = this.stream_manager.compute(j % this.stream_manager.size, compute_layer, i, j, layers_weights_sync[j], layers_cache_sync[j])
        else:
          # GPU attention, MLP, input, or output layer
          compute_f = this.stream_manager.compute(j % this.stream_manager.size, compute_layer, i, j, layers_weights_sync[j], layers_cache_sync[j])
          # next layer's weight & cache
          if j + 1 < this.num_layers and layers_weights_sync[j + 1] is None:
            f = this.cache_loader.load_cache((j + 1) % this.cache_loader.size, load_layer_weight, i, j + 1, this.cpu_del[j + 1])
            layers_weights_sync[j + 1] = f
            if this.layers[j + 1].need_cache:
              f = this.cache_loader.load_cache((j + 1) % this.cache_loader.size, load_layer_cache, i, j + 1, 0, this.cpu_del[j + 1])
              layers_cache_sync[j + 1] = f
          
        # wait for compute
        wait_stream_finish(compute_f)
    
    timers("generate").stop()

  
  def generation_loop_overlap_multi_batch(self, this, profile_dir):
    raise NotImplementedError()
