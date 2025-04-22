from abc import abstractmethod
from typing import List, Dict

import numpy as np
import gymnasium
import pathlib
from stable_baselines3.common.vec_env import DummyVecEnv

from algos import AbstractComponent
from algos.logger import DatabaseLogger
from ...abstractbaseclasses import AbstractRLComponent
from ...util.removetrunc import remove_truncated_if_needed
from ...defaultnamespace import EPOCH, TERMINAL, STEP, TRUNCATE, INFO, ENVREWARD


class GoalSampler:
    
    def __init__(self, goal_positions: List[np.ndarray], *args, **kwargs):
        self.goal_positions = goal_positions
        self._step = 0

    def sample(self, _):
        return self.goal_positions[self._step].copy()

    def increment_step(self, i):
        self._step = i 

    def reset(self):
        self._step = 0

class GoalEvaluatorComponent(AbstractRLComponent):
    default_io_map = {
        'epoch': EPOCH,
        'terminal': TERMINAL,
        'exp_step': STEP,
        'truncate': TRUNCATE,
        'info': INFO,
        'reward': ENVREWARD
    }
    def __init__(self, env:gymnasium.Env, last_step:int, io_map: Dict[str,str] = default_io_map, frequency: int = 100, *args, **kwargs):
        remove_truncated_if_needed(self, env, io_map)
        super().__init__(io_map, *args, **kwargs)
        self.env = env
        self.last_checked = 0
        self._env_stem = self.env.envs[0] if isinstance(env, DummyVecEnv) else self.env
        goal_positions = self.get_goal_positions()
        self.num_goals = len(goal_positions)
        self._sampler = GoalSampler(goal_positions)
        self._last_step = last_step
        self._evaluating = False
        self._frequency = frequency
        self._training_sampler = None
        self._evaluating_position = 0
        self._evaluated_all = 0
        self._info_store = None
        self._return = 0
        self._init_pos = None
        self._rl_env = None

    def set_rl_env(self, rl_env):
        self._rl_env = rl_env

    def set_init_pos(self, init_pos):
        self._init_pos = init_pos

    def step(self):
        self._step = self.exp_step

    def update(self):
        terminal = self.terminal
        info = self.info
        reward = self.reward
        self._return += reward[0]
        if terminal:
            epoch = self.epoch
            self._info_store = info
    
    def evaluate(self):
        if self.will_evaluate(self._step) and not self._evaluating:
            self._evaluating = True
            AbstractRLComponent._evaluating = True
            AbstractComponent._is_training = False
            DatabaseLogger._is_training = False
            print(f"EVALUATING@{self._step}")
            self.last_checked = self._step
            self._training_sampler = self._env_stem.get_sampler()
            self._env_stem.set_evaluate()
            self._env_stem.set_sampler(self._sampler)
            self._sampler.increment_step(self._evaluating_position)
        elif self._evaluating:
            print(f'Evaluation: {self._evaluating_position + (self._evaluated_all * self.num_goals)}')
            success = self._info_store[0].get("success", None)
            if success is None:
                success = self.reward > 0
            DatabaseLogger._instance.record_evaluation(init_position = self._init_pos['observation'],
                                                       goal_value = self._sampler.sample(self._evaluating_position),
                                                       reward = self._return, 
                                                       success = success,
                                                       step = self.last_checked)
            self._evaluating_position += 1
            if self._evaluating_position >= self.num_goals:
                self._evaluating_position = 0
                self._evaluated_all += 1
                if self._evaluated_all >= 1:
                    print("FINISHED EVALUATION")
                    self._env_stem.unset_evaluate()
                    self._env_stem.set_sampler(self._training_sampler)
                    AbstractComponent._is_training = True
                    DatabaseLogger._is_training = True
                    AbstractRLComponent._evaluating = False
                    self._evaluating = False
                    self._evaluated_all = 0
                    self._evaluating_position = 0
                    self._training_sampler = None
                    self._rl_env._exp_step._state = self.last_checked
            else:
                self._sampler.increment_step(self._evaluating_position)
            
    def will_evaluate(self, check_num: int) -> bool:
        return (check_num % self._frequency == 0) or ((check_num - self.last_checked) > self._frequency) or (check_num >= self._last_step)
                
    def evaluation_loop(self, check_num):
        for _ in range(4):
            for i in range(self.num_goals):
                episode_reward = 0
                terminated = terminate =False
                self._env_stem.get_sampler().increment_step(i)
                obs = self.env.reset()
                init_obs = obs
                while not (terminate):    
                    action, _ = self.model.predict(obs, deterministic=True)    
                    obs, reward, terminated, info = self.env.step(action)    
                    episode_reward += float(reward)    
                    if terminated or info[0].get("success", False):
                        success = info[0].get("success", None)
                        if success is None:
                            success = episode_reward > 0
                        DatabaseLogger._instance.record_evaluation(init_position = init_obs['observation'],
                                            goal_value = self._sampler.sample(i),
                                            reward = episode_reward, 
                                            success = success,
                                            step = check_num)
                        self._env_stem.env.goal = None
                        terminate = True

    def get_goal_positions(self)->List[np.ndarray]:
        if self._env_stem.spec.id == "DCMotor-v0":
            return self.get_dcmotor_goal_positions()
        elif hasattr(self._env_stem,"maze"):
            return self.get_maze_goal_positions()
        else:
            raise NotImplementedError("GoalEvaluator not implemented for this environment")

    def get_maze_goal_positions(self) -> List[np.ndarray]:
        return self._env_stem.maze.unique_goal_locations
    
    def get_dcmotor_goal_positions(self) -> List[np.ndarray]:
        goal_space = self._env_stem.observation_space['desired_goal']
        return np.arange(goal_space.low, goal_space.high, (goal_space.high - goal_space.low)/10.0).reshape(-1,1)
    
    def initialise(self):
        self._return = 0

    @classmethod
    def set_up_hyperparameters(cls):
        pass

    def load(self):
        pass

    def save(self):
        pass