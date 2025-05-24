import multiprocessing as mp
from functools import partial
import sys

from dynagen_optimize import DynagenOptOverlappingHeuristic, DynagenOptWorksetHeuristic
from network_config import Llama13BConfig, Llama70BConfig

gpu_mem = 8
prompt_len = 1024
gen_len = 64
gbs = 4
b = 4
tol = 15

def process(profiler, position, weight_percent, cache_percent):
    try:
        opt = DynagenOptOverlappingHeuristic(len(profiler.get_weights()), gbs, b, prompt_len, gen_len, gpu_mem, profiler, cost_tolerance=float(tol))
        # opt = DynagenOptWorksetHeuristic(len(profiler.get_weights()), gbs, b, prompt_len, gen_len, gpu_mem, profiler)
        cost, *policy = opt.optimize_policy(weight_percent=weight_percent, cache_percent=cache_percent, position=position)
        return weight_percent, cache_percent, cost
    except Exception as e:
        # print(e)
        return None

if __name__ == "__main__":
    with open(f'dynagen_optimize_overlapping_70b_{gpu_mem}g_{tol}xtol_test_output.csv', 'w') as f:
    # with open(f'dynagen_optimize_workset_70b_{gpu_mem}g_test_output.csv', 'w') as f:
        f.write("Weight percent,Cache percent,Cost\n")
        llama_config = Llama70BConfig()
        percents = [(w, c) for w in range(100, -1, -1) for c in range(100, -1, -1)]

        if 'pydevd' in sys.modules:
            num_processes = 1
        else:
            num_processes = 16
        worker_func = partial(process, llama_config)

        with mp.Pool(processes=num_processes) as pool:
            tasks = [((i % num_processes) + 1, w, c) for i, (w, c) in enumerate(percents)]
            results = pool.starmap(worker_func, tasks)

            for result in results:
                if result is not None:
                    weight_percent, cache_percent, cost = result
                    # print(f"wg = {weight_percent}%, cg = {cache_percent}%")
                    f.write(f"{weight_percent},{cache_percent},{cost}\n")
                    f.flush()
