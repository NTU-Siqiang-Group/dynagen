import numpy as np
from scipy.optimize import minimize

from flexgen.optimize.network_config import ProfilerConfig

class NetworkProfiler:
    def __init__(self, htod_cost, dtoh_cost, prefill_batch, compute_cache_gpu, compute_cache_cpu, compute_mlp_gpu):
        self.htod_cost = htod_cost
        self.dtoh_cost = dtoh_cost
        self.prefill_batch = prefill_batch
        self.compute_cache_gpu = compute_cache_gpu
        self.compute_cache_cpu = compute_cache_cpu
        self.compute_mlp_gpu = compute_mlp_gpu

    def __call__(self):
        def memory_cost_function(params):
            htod_cost, dtoh_cost = params

            htod_loss = np.sum((htod_cost - self.htod_cost)**2)
            dtoh_loss = np.sum((dtoh_cost - self.dtoh_cost)**2)
            total_loss = htod_loss + dtoh_loss
            return total_loss

        def compute_loss_function(params):
            prefill_batch, compute_cache_gpu, compute_cache_cpu, compute_mlp_gpu = params
            
            prefill_batch_loss = np.sum((prefill_batch - self.prefill_batch)**2)
            cache_gpu_loss = np.sum((compute_cache_gpu - self.compute_cache_gpu)**2)
            cache_cpu_loss = np.sum((compute_cache_cpu - self.compute_cache_cpu)**2)
            mlp_gpu_loss = np.sum((compute_mlp_gpu - self.compute_mlp_gpu)**2)
            
            total_loss = prefill_batch_loss + cache_gpu_loss + cache_cpu_loss + mlp_gpu_loss
            
            return total_loss

        # result = minimize(
        #     memory_cost_function, 
        #     [self.htod_cost[0], self.dtoh_cost[0]],
        #     method='L-BFGS-B',
        #     bounds=[(1e-15, 1e-8), (1e-15, 1e-8)],
        # )
        # htod_cost, dtoh_cost = result.x
        htod_cost = np.mean(self.htod_cost)
        dtoh_cost = np.mean(self.dtoh_cost)

        result = minimize(
            compute_loss_function, 
            [self.prefill_batch[0], self.compute_cache_gpu[0], self.compute_cache_cpu[0], self.compute_mlp_gpu[0]], 
            method='L-BFGS-B',
            bounds=[(1e-4, 1e-1), (1e-5, 1e-1), (1e-5, 1e-1), (1e-5, 1e-1)],
        )
        prefill_batch, compute_cache_gpu, compute_cache_cpu, compute_mlp_gpu = result.x

        ProfilerConfig.htod_cost = float(format(htod_cost, '.3g'))
        ProfilerConfig.dtoh_cost = float(format(dtoh_cost, '.3g'))
        ProfilerConfig.prefill_batch = float(format(prefill_batch, '.1g'))
        ProfilerConfig.compute_cache_gpu = float(format(compute_cache_gpu, '.1g'))
        ProfilerConfig.compute_cache_cpu = float(format(compute_cache_cpu, '.1g'))
        ProfilerConfig.compute_mlp_gpu = float(format(compute_mlp_gpu, '.1g'))

        print("ProfilerConfig(")
        print(f"\thtod_cost = {htod_cost:.3g},")
        print(f"\tdtoh_cost = {dtoh_cost:.3g},")
        print(f"\tprefill_batch = {prefill_batch:.1g},")
        print(f"\tcompute_cache_gpu = {compute_cache_gpu:.1g},")
        print(f"\tcompute_cache_cpu = {compute_cache_cpu:.1g},")
        print(f"\tcompute_mlp_gpu = {compute_mlp_gpu:.1g}")
        print(")")

