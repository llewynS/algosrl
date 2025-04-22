from abc import abstractmethod
from typing import Union, Dict, List
import types
from threading import Event
from algos import AbstractComponent, RegisterMeta, AbstractHyperParser

def concatenate_inputs(a: Union[Dict[str,str],List[str]], b: Union[Dict[str,str],List[str]])->Union[Dict[str,str],List[str]]:
    """Concatenate two inputs 

    :param a: input a
    :type a: Union[Dict[str,str],List[str]]
    :param b: input b
    :type b: Union[Dict[str,str],List[str]]
    :raises TypeError: _description_
    :return: both input a and b
    :rtype: Union[Dict[str,str],List[str]]
    """    
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, dict) and isinstance(b, dict):
        return {**a, **b}
    else:
        raise TypeError("Inputs must be of the same type (list or dict)")
    
class AbstractRegisteredHyperParser(AbstractHyperParser, RegisterMeta):
    pass

class AbstractRLComponent(AbstractComponent):
    """Abstract RL Component defines the functions that will be common to all RL Components

    All components used for RL will need to define at least initialise, step, and update. These can simply be pass if 
    not required.

    The intention of the three functions are as follows:

    initialise: Kick off the components. Defines initial conditions x[0] and will set the subject's values who start the algorithm. e.g call env.reset() to get the first state
    step: Essentially a forward pass of the components. e.g agent selects action based on state, env takes action and generates next state and reward. 
    update: Runs any learning required. e.g updates the agents policy and/or value function. 

    If additional algorithmic steps are required for customised RL algorithms one can decompose any of the three functions
    into additional required subfunctions. e.g step could call stepA, stepB, and stepC 

    It is important to note that all control is dictated by the observer - subject relationship defined in OS. The 
    algorithms will be self organising based on inputs (observers) and outputs (subjects). There is currently no 
    checking in place to detect that IO has been conditioned correctly. It is possible to hang on a function
    if the IO is not appropriately defined. 

    Terminal is an abstract property of all RL Components as it defines the stopping condition
    """
    abstract_methods = ["initialise", "step", "update"]
    max_epochs = None
    max_steps = None
    _all_rollouts_finished = Event()
    _num_in_rollout = 0
    _evaluating = False

    def __init__(self, io_map: Dict[str, str], additional_io_methods:List[str]=['rollout'], *args, **kwargs):
        self._component_collection = None
        super().__init__(io_map, additional_io_methods = additional_io_methods, *args, **kwargs)

    def do_run(self):
        self.init_epoch()
        while self.continue_run():
            self.rollout()
            # self._reset_subjects()
        self.save()

    def rollout(self):
        AbstractRLComponent._num_in_rollout += 1
        self.initialise()
        while(not (self.terminal or self.truncated)):
            self.step()
            self.update()
        self.done_rollout()
        self._all_rollouts_finished.wait()
        self._all_rollouts_finished.clear()
        self._reset_subjects()

    def init_epoch(self):
        pass

    def evaluate(self):
        pass

    def continue_run(self):
        if self.max_epochs is None and self.max_steps is None:
            raise AttributeError("Either max_epochs or max_steps must be defined")
        epoch = self.epoch
        if self._evaluating:
            return True
        if epoch%1000 == 0:
            self.save()
        if self.max_epochs is None:
            return self.exp_step < self.max_steps
        elif self.max_steps is None:
            return epoch < self.max_epochs
        else:
            return epoch < self.max_epochs or self.exp_step < self.max_steps
            
    def done_rollout(self):
        if hasattr(self._epoch, '_accessed'):
            self._epoch._accessed = self._epoch._num_access
        # print(self._num_in_rollout)
        try:
            self.evaluate()
            AbstractRLComponent._num_in_rollout -= 1
            if self._num_in_rollout == 0:
                self._all_rollouts_finished.set()
        except Exception as e:
            self._all_rollouts_finished.set()
            raise e

    def _reset_subjects(self):
        for _, subject in self.io.subjects.items():
            for observer in subject._observers:
                observer._accessed = observer._num_access