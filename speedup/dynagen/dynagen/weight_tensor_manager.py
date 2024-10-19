import os
import re
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from flexgen.pytorch_backend import DeviceType, TorchDevice, TorchDisk, TorchTensor, general_copy
from flexgen.utils import ValueHolder, torch_dtype_to_np_dtype


class TorchCPUWeightTensorManager:
    DUMMY_WEIGHT = "_DUMMY_"

    def __init__(self, env):
        self._cpu_weights: Dict[str | Tuple, TorchTensor] = {}
        self._cpu_weight_sizes: List[Tuple[str, int]] = []
        self._cpu_weights_initialized = False

        self._cpu_compressed_weights: Dict[str | Tuple, TorchTensor] = {}
        self._cpu_compressed_weight_sizes: List[Tuple[str, int]] = []
        self._cpu_compressed_weights_initialized = False

        assert isinstance(env.cpu, TorchDevice)
        self._dev: TorchDevice = env.cpu
        self._comp_dev = self._dev.compressed_device
        self._dev_choices: Tuple[TorchDisk, TorchDevice, TorchDevice] = (env.disk, env.cpu, env.gpu)

    def _get_choice(
        self,
        dummy: bool,
        key: str,
        compress: bool,
        percents: List[float],
        cur_percent: Optional[float] = None,
    ) -> TorchDevice:
        percents = np.cumsum(percents)
        assert np.abs(percents[-1] - 100) < 1e-5

        if dummy:
            assert cur_percent is not None
            for i in range(len(percents)):
                if cur_percent < percents[i]:
                    return self._dev_choices[i]
            return self._dev_choices[-1]

        key = os.path.basename(key)
        if not compress:
            cur_idx = self._cpu_weight_percent_cumsum_idx[key]
            boundaries = np.searchsorted(self._cpu_weight_percent_cumsum["size"], percents)
        else:
            cur_idx = self._cpu_compressed_weight_percent_cumsum_idx[key]
            boundaries = np.searchsorted(self._cpu_compressed_weight_percent_cumsum["size"], percents)

        for i in range(len(boundaries)):
            if cur_idx < boundaries[i]:
                return self._dev_choices[i]
        return self._dev_choices[-1]

    def _init_cpu_weight_percent_cumsum(self, compress: bool):
        if (not compress and self._cpu_weights_initialized) or (compress and self._cpu_compressed_weights_initialized):
            return

        cpu_weight_sizes = self._cpu_weight_sizes if not compress else self._cpu_compressed_weight_sizes
        cpu_weight_sizes_np = np.array(cpu_weight_sizes, dtype=[("key", "U50"), ("size", int)])
        cpu_weight_sizes_np.sort(order="size")
        cpu_weight_sizes_sum = cpu_weight_sizes_np["size"].sum()
        cpu_weight_sizes_np["size"] = cpu_weight_sizes_np["size"].cumsum()
        cpu_weight_sizes_np["size"] = cpu_weight_sizes_np["size"] / cpu_weight_sizes_sum * 100

        if not compress:
            self._cpu_weight_percent_cumsum = cpu_weight_sizes_np
            self._cpu_weight_percent_cumsum_idx = {key: i for i, key in enumerate(cpu_weight_sizes_np["key"].tolist())}
            self._cpu_weights_initialized = True
        else:
            self._cpu_compressed_weight_percent_cumsum = cpu_weight_sizes_np
            self._cpu_compressed_weight_percent_cumsum_idx = {
                key: i for i, key in enumerate(cpu_weight_sizes_np["key"].tolist())
            }
            self._cpu_compressed_weights_initialized = True
        assert len(cpu_weight_sizes) == len(self._cpu_weight_percent_cumsum_idx)

    def _get_cpu_weight(self, key, compress, shape, dtype, comp_weight_config, tensor=None) -> TorchTensor:
        if not compress:
            cpu_weight = self._cpu_weights.get(os.path.basename(key))
            if cpu_weight is not None:
                if tensor is not None:  # Cache tensor only when not compressing the weights
                    key = os.path.basename(key)
                    idx = self._cpu_weight_sizes.index((key, cpu_weight.bytes))
                    cpu_weight.data = tensor
                    cpu_weight.shape = tensor.shape
                    self._cpu_weights[key] = cpu_weight
                    self._cpu_weight_sizes[idx] = (key, cpu_weight.bytes)
                return cpu_weight
        else:
            cpu_weight = self._cpu_compressed_weights.get(os.path.basename(key))
            if cpu_weight is not None:
                return cpu_weight

        # cpu_weights MISS, create and store it
        if self.DUMMY_WEIGHT not in key:  # Use real weights, key == filename
            filename = key
            key = os.path.basename(filename)
            if not compress:
                cpu_weight = self._dev.allocate(shape, dtype, pin_memory=True)
                cpu_weight.load_from_np_file(filename)
                self._cpu_weights[key] = cpu_weight
                self._cpu_weight_sizes.append((key, cpu_weight.bytes))
            else:
                cpu_weight = self._comp_dev.allocate(shape, dtype, comp_weight_config, pin_memory=True)
                cpu_weight.load_from_np_file(filename)
                self._cpu_compressed_weights[key] = cpu_weight
                self._cpu_compressed_weight_sizes.append((key, cpu_weight.bytes))
        else:  # Use dummy weights for benchmark purposes, key == shape
            if not compress:
                cpu_weight = self._dev.allocate(shape, dtype, pin_memory=True)
                cpu_weight.load_from_np(np.ones(shape, dtype))
                self._cpu_weights[key] = cpu_weight
                self._cpu_weight_sizes.append((str(key), cpu_weight.bytes))
            else:
                cpu_weight = self._comp_dev.allocate(shape, dtype, comp_weight_config, pin_memory=True)
                for i in range(2):
                    x = cpu_weight.data[i]
                    assert isinstance(x, TorchTensor)
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
                self._cpu_compressed_weights[key] = cpu_weight
                self._cpu_compressed_weight_sizes.append((str(key), cpu_weight.bytes))
        return cpu_weight

    def init_cpu_weight(self, weight_specs: List[Tuple], policy):
        dummy = self.DUMMY_WEIGHT in weight_specs[0][2]
        for weight_spec in weight_specs:
            shape, dtype, filename, *args = weight_spec

            if len(shape) < 2:
                compress = False
            else:
                compress = policy.compress_weight

            key = filename if not dummy else shape
            if len(args) == 0:
                self._get_cpu_weight(key, compress, shape, dtype, policy.comp_weight_config)
                continue

            before_init_cpu_weight_hook, kwargs = args
            assert isinstance(before_init_cpu_weight_hook, types.FunctionType)
            assert isinstance(kwargs, dict)

            for k, maybe_weights_idx in kwargs.items():
                if not isinstance(maybe_weights_idx, str):
                    continue

                matched = re.search(r"weights\[(\d+)\]", maybe_weights_idx)
                if not matched:
                    continue

                shape, dtype, filename, *_ = weight_specs[
                    int(matched.group(1))
                ]  # No support for resolving recursive after_init_cpu_weight_hook for now
                kwargs[k] = self._get_cpu_weight(filename, compress, shape, dtype, policy.comp_weight_config)

            tensor = before_init_cpu_weight_hook(**kwargs)
            assert isinstance(tensor, torch.Tensor)
            self._get_cpu_weight(key, compress, shape, dtype, policy.comp_weight_config, tensor)

    def set_weight_home(
        self, weight_home: ValueHolder, weight_specs: List[Tuple], weight_read_buf: ValueHolder, policy
    ):
        self._init_cpu_weight_percent_cumsum(policy.compress_weight)
        dev_percents = (policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent)
        dummy = self.DUMMY_WEIGHT in weight_specs[0][2]

        if dummy:
            sizes = [np.prod(spec[0]) for spec in weight_specs]
            sizes_cumsum = np.cumsum(sizes)

        for i in range(len(weight_specs)):
            shape, dtype, filename, *_ = weight_specs[i]
            key = filename if not dummy else shape

            if dummy:
                mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1] * 100
                home = self._get_choice(dummy, key, policy.compress_weight, dev_percents, mid_percent)
            else:
                home = self._get_choice(dummy, key, policy.compress_weight, dev_percents)

            if weight_home.val[i] is not None:
                if home.device_type == weight_home.val[i].device.device_type:
                    # The weight is already loaded on the correct device, skip
                    continue

                if weight_read_buf.val is not None:
                    (buffered_weight, _) = weight_read_buf.val[i]
                    if buffered_weight is not None and home.device_type == buffered_weight.device.device_type:
                        # The weight is already buffered on the correct device, append it and continue
                        weight_home.val[i] = buffered_weight
                        continue

                if weight_home.val[i].device is not None and weight_home.val[i].device.device_type != DeviceType.CPU:
                    # The weight is loaded on a different device other than CPU, delete it
                    weight_home.val[i].delete()

            if len(shape) < 2:
                pin_memory = True
                compress = False
            else:
                pin_memory = policy.pin_weight
                compress = policy.compress_weight

            cpu_weight = self._get_cpu_weight(key, compress, shape, dtype, policy.comp_weight_config)

            if home.device_type == DeviceType.CPU:
                # CPU is chosen, append cpu_weight and continue
                weight_home.val[i] = cpu_weight
                continue

            if not compress:
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)
            else:
                weight = home.compressed_device.allocate(shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if home.device_type == DeviceType.DISK:
                # Disk is chosen, directly copy the weight file
                weight.load_from_np_file(filename)
            else:
                # GPU is chosen, copy from cpu_weight
                general_copy(weight, None, cpu_weight, None)
            weight_home.val[i] = weight
