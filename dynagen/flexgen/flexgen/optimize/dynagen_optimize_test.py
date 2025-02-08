from dynagen_optimize import DynagenOptWorksetHeuristic
from network_config import Llama1BConfig, Llama13BConfig

def summarize_policy(gen_len, num_layers, num_batches, opt):
    cache, weight, cpu_del = opt.get_policy()
    mem_consumption = opt.get_mem_consumption_full(
        opt.cache_prefetch, opt.weight_prefetch, opt.cpu_delegation, opt.weight_percent, opt.cache_percent
    )
    print('token|layer|batch|fetch_cache|fetch_weight|use_cpu_del|memory(MB)')
    idx = 0
    for i in range(gen_len):
        for j in range(num_layers):
            for k in range(num_batches):
                fetch_cache = 'No'
                if (i, j, k) in cache:
                    fetch_cache = cache[(i, j, k)]
                fetch_weight= 'No'
                if (i, j, k) in weight:
                    fetch_weight = weight[(i, j, k)]
                print(f'{i}|{j}|{k}|{fetch_cache}|{fetch_weight}|{cpu_del[(i, j, k)]}|{mem_consumption[idx] / (1<<20)}')
                idx += 1

if __name__ == "__main__":
    llama_config = Llama13BConfig()
    opt = DynagenOptWorksetHeuristic(len(llama_config.get_weights()), 8, 8, 1024, 64, 20, llama_config)
    wg, cg = opt.optimize()
    print(f"Optimized weight_gpu_percent: {wg}, cache_gpu_percent: {cg}")
