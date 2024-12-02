from dynagen_optimize import DynagenOpt
from network_config import Llama1BConfig

def summarize_policy(gen_len, num_layers, num_batches, opt):
    cache, weight, cpu_del = opt.get_policy()
    mem_consumption = opt.get_mem_consumption_full(
        opt.cache_prefetch, opt.weight_prefetch, opt.cpu_delegation
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
    llama_config = Llama1BConfig()
    opt = DynagenOpt(len(llama_config.get_weights()), 4, 16, 512, 32, llama_config)
    opt.optimize(20)
    # summarize_policy(32, len(llama_config.get_weights()), 16, opt)
