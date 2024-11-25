from flexgen.computation_policy_default import ComputationPolicyImpl
from flexgen.computation_policy_streams import ComputationPolicyStream
from flexgen.computation_policy_interface import ComputationPolicyInterface
from flexgen.computation_policy_alter_stream import ComputationPolicyAlterStream
from flexgen.computation_policy_alter_v2 import ComputationPolicyAlterStreamV2

def get_computation_policy(choice='default'):
  if choice == 'default':
    print("Using default computation policy")
    return ComputationPolicyImpl()
  elif choice == 'stream':
    print("Using stream computation policy")
    return ComputationPolicyStream()
  elif choice == 'alter_stream':
    print("Using alter stream computation policy")
    return ComputationPolicyAlterStream()
  elif choice == 'alter_v2':
    print("Using alter v2 computation policy")
    return ComputationPolicyAlterStreamV2()
  
  return ComputationPolicyInterface()