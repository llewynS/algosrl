from algos import AbstractComponentFactory, get_class_args
from ..goal import UniformSamplerNotComponent

from abc import abstractmethod

class GoalFactory(AbstractComponentFactory):
    _register = []
    _is_env_sampler = True
    def build(self):
        sampler = self.build_sampler()
        if self._is_env_sampler:
            if hasattr(self._experiment._env, 'envs'):
                self._experiment._env.envs[0].set_sampler(sampler)
            else:
                self._experiment._env.set_sampler(sampler)

    @abstractmethod
    def build_sampler(self):
        pass

class UniformFactory(GoalFactory):
    _register = [UniformSamplerNotComponent]

    def build_sampler(self):
        return UniformSamplerNotComponent(self._experiment._env, **get_class_args(UniformSamplerNotComponent,self._experiment._exp_args))
