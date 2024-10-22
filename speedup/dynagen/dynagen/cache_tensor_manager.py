from flexgen.pytorch_backend import DeviceType, TorchMixedDevice, TorchTensor, general_copy, SEG_DIM
from flexgen.utils import np_dtype_to_torch_dtype


class TorchCacheTensor(TorchTensor):
    def __init__(self, shape, dtype, data, device, name=None):
        assert isinstance(device, TorchCacheTensorDevice)
        super().__init__(shape, dtype, data, device, name)

        self._cpu_tensor = self.copy(device.cpu)
        self._buf_indices = None

    def store_cache(self, indices, new_cache):
        '''
        Store the new cache tensor and update the _cpu_tensor.
        '''
        self._buf_indices = indices
        general_copy(self, indices, new_cache, None)

        if new_cache.device.device_type == DeviceType.CPU:
            self._cpu_tensor = new_cache
        else:
            general_copy(self._cpu_tensor, indices, new_cache, None)

    def set_cache_percents(self, policy, num_head: int, cache_write_buf: TorchTensor):
        '''
        Set the cache percents (seg_lengths) to the values specified in the policy.
        '''
        shape = self.shape
        seg_lengths = self.device.get_seg_lengths(policy, self.shape, num_head) # (len_gpu, len_cpu, len_disk)
        is_using_buf = cache_write_buf and cache_write_buf.device and cache_write_buf.device.device_type == DeviceType.CUDA

        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.device.base_devices
        tensors = [None] * len(devices)
        for i in range(len(devices) - 1, -1, -1):
            if self.data[1][i : i + 1] == seg_points[i : i + 1]:
                # The new segment is the same as the existing one, append the existing segment
                tensors[i] = self.data[0][i]
                continue

            self.data[0][i].delete()
            device = devices[i]
            seg_len = seg_points[i + 1] - seg_points[i]
            if seg_len == 0:
                tensors[i] = None
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM + 1 :]
                src_indices = (slice(0, seg_shape[0]), slice(seg_points[i], seg_points[i + 1]))

                if device.device_type == DeviceType.CUDA and is_using_buf and tuple(cache_write_buf.shape) == seg_shape:
                    # TODO: upload from cache_write_buf
                    tensor = cache_write_buf.data[src_indices]
                    tensor = TorchTensor.create_from_torch(tensor, cache_write_buf.device)
                elif device.device_type == DeviceType.CPU:
                    tensor = self._cpu_tensor.data[src_indices]
                    tensor = TorchTensor.create_from_torch(tensor, self._cpu_tensor.device)
                else:
                    tensor = self._cpu_tensor.copy(device, src_indices)
                tensors[i] = tensor

        self.data = (tensors, seg_points)


class TorchCacheTensorDevice(TorchMixedDevice):
    def __init__(self, base_devices):
        super().__init__(base_devices)

        for dev in base_devices:
            if dev.device_type == DeviceType.CPU:
                self.cpu = dev
                break
        self.device_type = DeviceType.CACHE_MIXED

    def get_seg_lengths(self, policy, shape, num_head):
        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM] - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        return (len_gpu, len_cpu, len_disk)

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i + 1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM + 1 :]
                tensors.append(
                    devices[i].allocate(seg_shape, dtype, pin_memory=pin_memory)
                )

        return TorchCacheTensor(
            shape,
            np_dtype_to_torch_dtype[dtype],
            (tensors, seg_points),
            self,
            name=name,
        )
