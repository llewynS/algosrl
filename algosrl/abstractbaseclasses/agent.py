from collections.abc import Callable
from typing import Dict


from .rlcomponent import AbstractRLComponent
from ..defaultnamespace import EPOCH, TERMINAL, ACT, ENVOBS, ENVOBSPRIME, ENVREWARD, TRUNCATE, INFO, STEP
from algos import AbstractParametered, AbstractHyperParser

class AbstractPolicy(AbstractParametered, Callable,
                     metaclass=AbstractHyperParser):
    """
    Abstract class for a policy.

    An agent and a policy could be considered synonymous but this configuration enables agents that use value functions

    Must be callable.
    """
    pass


class AbstractValueFunction(AbstractParametered, Callable,
                            metaclass=AbstractHyperParser):
    """
    Abstract Value function.

    Must be callable.
    """
    pass

class AbstractAgent(AbstractRLComponent):
    """
    An Abstract Agent Class
    """
    default_io_map = {"observation": ENVOBS,
                      "observation_prime": ENVOBSPRIME,
                      "reward": ENVREWARD,
                      "terminal": TERMINAL,
                      "action": ACT,
                      "epoch": EPOCH,
                      "truncated": TRUNCATE,
                      "exp_step":STEP,
                      "info": INFO}
    
    def __init__(self,
                 policy: AbstractPolicy,
                 value_function: AbstractValueFunction = None, 
                 io_map: Dict[str, str] = default_io_map,
                 *args, **kwargs):
        if policy is None:
            raise AttributeError("A policy must be given to the agent")
        self._policy = policy
        self._value_function = value_function
        super().__init__(io_map, *args, **kwargs)

        # self.return_ = 0.0

    @property
    def policy(self) -> AbstractPolicy:
        """Agent policy getter (read-only)

        :return: The agents policy
        :rtype: Policy
        """
        return self._policy