from .util import get_goal_bounds
import numpy as np
from torch import tensor as torch_tensor
from torch import float32 as torch_float32
from torch.distributions import Uniform
from .torch.targetdistributions import generate_target_distribution_maze
from .torch import device

from algos import AbstractParametered

class UniformSamplerNotComponent(AbstractParametered):
    def __init__(self, env, use_empty=0, use_start=0):
        self.env = env
        self.shape = env.observation_space['desired_goal'].shape
        self.bounds = get_goal_bounds(env)
        if hasattr(env, 'envs'):
            env = env.envs[0]
        if hasattr(env, 'maze'):
            self._goal_distribution = generate_target_distribution_maze(env, use_empty_cells=bool(use_empty), use_start_cells=bool(use_start))
        else:
            self._goal_distribution = Uniform(low=torch_tensor(self.bounds[0], dtype=torch_float32).to(device), 
                                              high=torch_tensor(self.bounds[1], dtype=torch_float32).to(device)) 

    def sample(self, _):
        return self._goal_distribution.sample().cpu().numpy()
    
    @classmethod
    def set_up_hyperparameters(cls):
        pass