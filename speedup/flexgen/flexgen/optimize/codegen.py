def exec_code_gen(exec_len, num_layers, num_gpu_batches, dyn_opt):
    cache_prefetch, weight_prefetch, cpu_delegation = dyn_opt.get_policy()

    code = ''
    for i in range(exec_len):
        code += 'timers("generate").start()\n'
        for k in range(num_gpu_batches):
            code += f'this.update_attention_mask({i}, {k})\n'
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                cache_prefetches = []
                if (i, j, k) in cache_prefetch:
                    cache_prefetches = cache_prefetch[(i, j, k)]
                weight_prefetches = []
                if (i, j, k) in weight_prefetch:
                    weight_prefetches = weight_prefetch[(i, j, k)]
                for token, layer, batch in cache_prefetches:
                    cpu_del = cpu_delegation[(token, layer, batch)]
                    code += f'f = this.cache_loader.load_cache(True, load_layer_cache, {token}, {layer}, {batch}, {cpu_del})\n'
                    code += f'layers_cache_sync[{batch}][{layer}] = f\n'
                for token, layer, batch in weight_prefetches:
                    code += f'f = this.cache_loader.load_cache(True, load_layer_weight, {token}, {layer}, {batch})\n'
                    code += f'layers_weights_sync[{batch}][{layer}] = f\n'

                code += f'compute_layer({i}, {j}, {k}, layers_weights_sync[{k}][{j}], layers_cache_sync[{k}][{j}], layers_weights_sync, layers_cache_sync, {cpu_delegation[(i, j, k)]})\n'
                if i == 0:
                    code += 'this.sync()\n'
        code += 'timers("generate").stop()\n'
        code += 'print(len(timers("generate").costs), sum(timers("generate").costs) / len(timers("generate").costs))\n'

    return code