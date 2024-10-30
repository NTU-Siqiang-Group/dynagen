# Dynagen

```sh
git clone https://github.com/we1pingyu/dynagen
conda create -n infinigen python=3.9
conda activate infinigen
pip install -r requirements.txt
```


## Implement your own policy
There are two main policies that can effectively impact the performance:
1. The memory allocation policy
2. The computation policy

The first one is implemented in `speedup/flexgen/flexgen/pytorch_backend.py`

```python
class TorchMixedDeviceMemManager:
    """
    Interface for managing memory on mixed devices.
    """
    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        raise NotImplementedError()

    def delete(self, tensor):
        raise NotImplementedError()
    
    def init_cache_one_gpu_batch(self, config, task, policy):
        raise NotImplementedError()

def get_torch_mixed_device_mem_manager(choice='default', base_devices=[]) -> TorchMixedDeviceMemManager:
    if choice == 'default':
        return TorchMixedDevice(base_devices)
    return TorchMixedDeviceMemManager()
```

There are three functions to implememnt. You can refer to the default implementation, `TorchMixedDevice`, for flexgen in the same file.

The second policy determines the cache loading / offloading behaviour, invoked by the `generate` function.
```python
class ComputationPolicyInterface:
  """
  Computation policy interface
  """
  def generation_loop_normal(self, this, evaluate):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()

  def generation_loop_overlap_single_batch(self, this, evaluate):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_overlap_multi_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()

  def generation_loop_debug_single_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_debug_multi_batch(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
  
  def generation_loop_debug_normal(self, this):
    """
    Returns the number of batches to overlap in the generation loop
    """
    raise NotImplementedError()
```
The interface is defined in `speedup/flexgen/flexgen/computation_policy_interface.py` and the default implementation is provided in `speedup/flexgen/flexgen/computation_policy_default.py`. If you want to implement a new policy, you should create a new file named `compuation_policy_xxx.py` and then import your new method in `speedup/flexgen/flexgen/computation_policy.py`:
```python
def get_computation_policy(choice='default'):
  if choice == 'default':
    return ComputationPolicyImpl()
  elif choice == 'xxx':
    return ComputationPolicyXXX()
  return ComputationPolicyInterface()
```