from flexgen.computation_policy_default import ComputationPolicyImpl
from flexgen.computation_policy_streams import ComputationPolicyStream
from flexgen.computation_policy_interface import ComputationPolicyInterface
from flexgen.computation_policy_alter_stream import ComputationPolicyAlterStream

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
  
  return ComputationPolicyInterface()