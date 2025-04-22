import gymnasium
import time
from typing import Union, Dict, List

from algos import check_io_map_contains, AbstractComponent
from ..abstractbaseclasses import AbstractRLComponent
from ..defaultnamespace import ENVOBS, ENVREWARD, ACT, TERMINAL, ENVOBSPRIME, EPOCH, INFO, TRUNCATE, STEP
from ..util import remove_truncated_if_needed

OLD_API = {"action": ACT, 
            "observation_prime": ENVOBSPRIME, 
            "reward": ENVREWARD, 
            "terminal": TERMINAL, 
            "epoch": EPOCH, 
            "observation": ENVOBS,
            "info": INFO,
            "exp_step": STEP}

NEW_API = {**OLD_API, **{"truncated": TRUNCATE}}

class StandardRLEnv(AbstractRLComponent):
    default_io_map = NEW_API

    @check_io_map_contains(*list(default_io_map.keys()))
    def __init__(self, 
                 env: gymnasium.Env,
                 io_map: Dict[str, str] = default_io_map,
                 *args, **kwargs):
        self._env = env
        remove_truncated_if_needed(self, self._env, io_map)
        super().__init__(io_map, *args, **kwargs)
        self._evaluator = None
        

    def initialise(self):
        if "truncated" not in self._io_map.keys():
            obs = self._env.reset()
            info = None
        else:
            obs, info = self._env.reset()
        if isinstance(obs, dict):
            goal = obs["desired_goal"]
            self._logger_inst.record("goal", str(goal),ignore_frequency=True)
            if not AbstractComponent._is_training and self._evaluator is not None:
                self._evaluator.set_init_pos(obs)
        self._terminal.initialise_state(False, True)
        self._epoch.initialise_state(self._epoch._state, False)
        self._exp_step.initialise_state(self._exp_step._state, False)
        self._observation.initialise_state(obs)
        if "truncated" in self._io_map.keys():
            self._truncated.initialise_state(False, True)
        self._info.initialise_state(info, True)
        

    def step(self):
        action = self.action #$
        if "truncated" not in self._io_map.keys():
            self.observation_prime, self.reward, terminal, self.info = self._env.step(action)
        else:
            self.observation_prime, self.reward, terminal, self.truncated, self.info = self._env.step(action)

        self.exp_step = self.exp_step + 1
        # print("env: ", self.exp_step)
        self.terminal = terminal
        if terminal or self.truncated:
            self.epoch = self.epoch + 1
        

    def update(self):
        self.observation = self._observation_prime._state
    
    @classmethod
    def set_up_hyperparameters(cls):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def init_epoch(self):
        self._epoch.initialise_state(0, False)
        self._exp_step.initialise_state(0, False)
    
    def done(self):
        super().done()
        for obs in self._epoch._observers:
            obs._accessed -= 1
        for obs in self._exp_step._observers:
            obs._accessed -= 1