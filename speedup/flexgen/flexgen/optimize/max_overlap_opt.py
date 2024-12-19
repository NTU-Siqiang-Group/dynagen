import math
from flexgen.optimize.network_config import Llama13BConfig
from tqdm import tqdm

config = Llama13BConfig()

Batch = 8
Layer = 32
Seq_len = 8


def cost_prefetch_weight(start, w1):
    cost = 0
    for i in range(start, int(w1)):
        if i % Batch == 0:
            if i // Batch % 2 == 0:
                cost += config.get_htod_cost(config.get_mlp_size()) * 0.8
            else:
                cost += config.get_htod_cost(config.get_attn_size()) * 0.8
    return cost


def cost_prefetch_cache(start, c):
    cost = 0
    for i in range(start, int(c)):
        cost += config.get_htod_cost(config.get_cache_size(Batch, 2048))
    return cost


def cpu_del(x):
    return 6e-3


def gpu_compute(start, x):
    print(start, int(x))
    cost = 0
    for i in range(start, int(x)):
        if i % 2 == 0:
            cost += 1e-4 * 2
        else:
            cost += 5e-3 * 2
    return cost


def objective(w, c):
    s = Batch * Layer * Seq_len
    cost = 0
    w_prev = 0
    c_prev = 0
    diff = 0
    while w_prev < s or c_prev < s:
        T_prefetch = (
            cost_prefetch_weight(w_prev, w_prev + w)
            + cost_prefetch_cache(c_prev, c_prev + c)
            + cpu_del(min(w_prev + w, c_prev + c))
        )
        print(w,c )
        T_compute = gpu_compute(min(w_prev, c_prev), min(w_prev + w, c_prev + c) - 1)
        cost += max(T_prefetch, T_compute)
        if w_prev < s:
            w_prev += w
        if c_prev < s:
            c_prev += c
        # diff += abs(T_prefetch - T_compute)
        print("T_prefetch:", T_prefetch, "T_compute:", T_compute)
    print("w:", w, "c:", c)
    return cost


w_range = range(1, 16)
c_range = range(1, 16)

best_w, best_c = 1, 1
best_diff = math.inf

for w in tqdm(w_range):
    for c in c_range:
        diff = objective(w, c)
        if diff < best_diff:
            best_diff = diff
            best_w = w
            best_c = c

print("Optimal w:", best_w)
print("Optimal c:", best_c)
print("Minimized cost:", best_diff)
