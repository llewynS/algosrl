
from algos import AbstractComponentFactory, get_class_args
from ..env import StandardRLEnv

class EnvFactory(AbstractComponentFactory):
    _register = [StandardRLEnv]
    def build(self):
        self._components['env'] = StandardRLEnv(self._experiment._env, **get_class_args(StandardRLEnv, self._experiment._exp_args))