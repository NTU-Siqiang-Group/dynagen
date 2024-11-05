from flexgen.computation_policy_default import ComputationPolicyImpl
from flexgen.computation_policy_alter import ComputationAlterPolicyImpl
from flexgen.computation_policy_interface import ComputationPolicyInterface


def get_computation_policy(choice="default"):
    if choice == "default":
        return ComputationPolicyImpl()
    elif choice == "alter":
        return ComputationAlterPolicyImpl()

    return ComputationPolicyInterface()
