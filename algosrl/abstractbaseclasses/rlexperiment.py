import pathlib
import gymnasium as gym
from abc import abstractmethod

from algos import AbstractExperiment
from .rlcomponent import AbstractRLComponent

class AbstractRLExperiment(AbstractExperiment):
    def __init__(self, file_path: pathlib.Path,
                 exp_args: dict, *args, **kwargs):
        self._exp_args = exp_args
        for key, val in self._exp_args.items():
            if 'file_path' in key or 'filepath' in key:
                self._exp_args[key] = file_path
        AbstractRLComponent.max_epochs = exp_args['epochs']
        AbstractRLComponent.max_steps = exp_args['max_steps']
        self.generate_env()
        super().__init__(file_path, exp_args, *args, **kwargs)
        self.resolve_codependencies()

            
    @classmethod
    def experiment_params(cls):
        
        super().experiment_params()
        cls._parser.add_argument('--env', default='CartPole-v1', type=str)
        cls._parser.add_argument('--max-steps', default=None, type=int)
        cls._parser.add_argument('--epochs', default=None, type=int)
        cls._parser.add_argument('--max-episode-steps', type=int, default=None)
        cls._parser.add_argument('--reward-type', type=str, default='dense')
        cls._parser.add_argument('--target-distribution', type=str, default='')

    @abstractmethod
    def generate_env(self):
        pass

    def resolve_codependencies(self):
        pass