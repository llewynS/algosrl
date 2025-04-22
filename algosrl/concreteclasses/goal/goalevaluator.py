from abc import abstractmethod
from typing import List

import numpy as np
import gymnasium
import pathlib
from stable_baselines3.common.vec_env import DummyVecEnv

from algos import AbstractComponent
from algos.logger import DatabaseLogger
from ...abstractbaseclasses import AbstractRLEvaluator, AbstractPolicy

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

class GoalEvaluator(AbstractRLEvaluator):
    def __init__(self, file_path: pathlib.Path, model:AbstractPolicy, env:gymnasium.Env, last_step:int, frequency: int = 100, *args, **kwargs):
        super().__init__(frequency, *args, **kwargs)
        self.env = env
        self.model = model
        self.file_path = file_path
        self.last_checked = 0
        self._env_stem = self.env.envs[0] if isinstance(env, DummyVecEnv) else self.env
        goal_positions = self.get_goal_positions()
        self.num_goals = len(goal_positions)
        self._sampler = GoalSampler(goal_positions)
        self._last_step = last_step
    
    @classmethod
    def set_up_hyperparameters(cls):
        pass

    def evaluate(self, check_num: int, *args, **kwargs):
        if self.will_evaluate(check_num):
            AbstractComponent._is_training = False
            print(f"EVALUATING@{check_num}")
            training_sampler = self._env_stem.get_sampler()
            self._env_stem.set_evaluate()
            self._env_stem.set_sampler(self._sampler)
            self.evaluation_loop(check_num)
            self._env_stem.unset_evaluate()
            self._env_stem.set_sampler(training_sampler)
            self.last_checked = check_num
            AbstractComponent._is_training = True
            
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