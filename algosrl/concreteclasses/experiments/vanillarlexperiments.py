import gymnasium as gym
import numpy as np


from ...util import MyDummyVecEnv
from ...abstractbaseclasses.rlexperiment import AbstractRLExperiment
from ..factories.agentfactory import SACFactory, TD3Factory
from ..factories.envfactory import EnvFactory


class VanillaSB3RLExperiment(AbstractRLExperiment):
    def generate_env(self):
        self._env = MyDummyVecEnv([lambda: gym.make(self._exp_args['env'])])

class SACVanillaSB3RLExperiment(VanillaSB3RLExperiment):
    _builders=[SACFactory,EnvFactory]

class TD3VanillaSB3RLExperiment(VanillaSB3RLExperiment):
    _builders=[TD3Factory,EnvFactory]