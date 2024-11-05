from flexgen.computation_policy_default import ComputationPolicyImpl
from flexgen.computation_policy_streams import ComputationPolicyStream
from flexgen.computation_policy_interface import ComputationPolicyInterface

def get_computation_policy(choice='default'):
  if choice == 'default':
    print("Using default computation policy")
    return ComputationPolicyImpl()
  elif choice == 'stream':
    print("Using stream computation policy")
    return ComputationPolicyStream()
  
  return ComputationPolicyInterface()