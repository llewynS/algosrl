import gymnasium as gym
import gym_dcmotor
import numpy as np
import gymnasium_robotics
from copy import deepcopy

from stable_baselines3.common.vec_env import DummyVecEnv

from ...abstractbaseclasses.rlexperiment import AbstractRLExperiment
from ...util.envadaptors import MazeEnvWrapper, ReduceDCMotorEnv, SetGoalWrapper, NormaliseEnv
from ...util import MyDummyVecEnv

gym.register_envs(gymnasium_robotics)

class SB3RLExperiment(AbstractRLExperiment):
    mazes = {
            "single": [[1, 1, 1, 1, 1, 1, 1],
                       [1, "g", "g", "r", "g", "g", 1],
                       [1, 1, 1, 1, 1, 1, 1]],
            "triple" : [[1, 1, 1, 1, 1, 1, 1],
                    [1, "g", "g", "g", "g", "g", 1],
                    [1, "g", "g", "r", "g", "g", 1],
                    [1, "g", "g", "g", "g", "g", 1],
                    [1, 1, 1, 1, 1, 1, 1]],
            "square":  [[1, 1, 1, 1, 1, 1, 1],
                        [1, "g", "g", "g", "g", "g", 1],
                        [1, "g", "g", "g", "g", "g", 1],
                        [1, "g", "g", "r", "g", "g", 1],
                        [1, "g", "g", "g", "g", "g", 1],
                        [1, "g", "g", "g", "g", "g", 1],
                        [1, 1, 1, 1, 1, 1, 1]],
                         "square_9x9": [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, "g", "g", "g", "g", "g", "g", "g", 1],
                       [1, "g", 0, 0, 0, 0, 0, "g", 1],
                       [1, "g", 0, 0, 0, 0, 0, "g", 1],
                       [1, "g", 0, 0, "r", 0, 0, "g", 1],
                       [1, "g", 0, 0, 0, 0, 0, "g", 1],
                       [1, "g", 0, 0, 0, 0, 0, "g", 1],
                       [1, "g", "g", "g", "g", "g", "g", "g", 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            "square_15x15": [
                                [1] * 15,
                                [1] + ["g"] * 13 + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 5 + ["r"] + [0] * 5 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] + [0] * 11 + ["g"] + [1],
                                [1] + ["g"] * 13 + [1],
                                [1] * 15
                            ],
            "square_21x21": [
                                [1] * 21,
                                [1] + ["g"] * 19 + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 8 + ["r"] + [0] * 8 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] + [0] * 17 + ["g"] + [1],
                                [1] + ["g"] * 19 + [1],
                                [1] * 21
                            ],                          
            "corner_up" : [[1, 1, 1, 1, 1, 1, 1],
                           [1, "r", "g", "g", "g", "g", 1],
                           [1, 1, 1, 1, 1, "g", 1],
                           [1, 1, 1, 1, 1, "g", 1],
                           [1, 1, 1, 1, 1, "g", 1],
                           [1, 1, 1, 1, 1, "g", 1],
                           [1, 1, 1, 1, 1, "g", 1],
                           [1, 1, 1, 1, 1, 1, 1]],
            "corner_down" : [[1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, "g", 1],
                             [1, 1, 1, 1, 1, "g", 1],
                             [1, 1, 1, 1, 1, "g", 1],
                             [1, 1, 1, 1, 1, "g", 1],
                             [1, 1, 1, 1, 1, "g", 1],
                             [1, "r", "g", "g", "g", "g", 1],
                             [1, 1, 1, 1, 1, 1, 1]],
            "u_maze" : [[1, 1, 1, 1, 1],
                        [1, "r", "g", "g", 1],
                        [1, 1, 1, "g", 1],
                        [1, "g", "g", "g", 1],
                        [1, 1, 1, 1, 1]],
            "u_maze_single" : [[1, 1, 1, 1, 1],
                        [1, "r", 0, 0, 1],
                        [1, 1, 1, 0, 1],
                        [1, "g", 0, 0, 1],
                        [1, 1, 1, 1, 1]],
            "u_maze_single_large" : [[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, "r", 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, "g", 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1 ,1 ,1]],
            "u_maze_bidirectional": [[1, 1, 1, 1, 1],
                        [1, "g", 0, 0,1],
                        [1, 1, 1, 0, 1],
                        [1, "r", 0, 0, 1],
                        [1, 1, 1, 0, 1],
                        [1, "g", 0, 0, 1],
                        [1, 1, 1, 1, 1]],
            "u_maze_bidirectional_large": [[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, "g", 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, "r", 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, "g", 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1 ,1 ,1]],
            "small_spiral" : [[1, 1, 1, 1, 1, 1],
                    [1, "r", "g", "g", "g", 1],
                    [1, 1, 1, 1, "g", 1],
                    [1, "g", "g", 1, "g", 1],
                    [1, "g", 1, 1, "g", 1],
                    [1, "g", "g", "g", "g", 1],
                    [1, 1, 1, 1, 1, 1]]                      
            }
    
    def generate_env(self):
        if "Maze" in self._exp_args['env']:
            self.generate_maze_env()
        elif "DCMotor" in self._exp_args['env']:
            self.generate_dc_motor_env()
        self._env = MyDummyVecEnv([lambda: self._env])


    def generate_dc_motor_env(self):
        sparse_scalar = self._exp_args.pop('sparse_scalar', 1)
        self._env = gym.make(self._exp_args['env'], max_episode_steps=self._exp_args['max_episode_steps'], 
                             reward_type=self._exp_args['reward_type'], continuing_task=self._exp_args['continuing_task'],
                             sparse_scalar=sparse_scalar, tolerance = 0.025)
        self._env = SetGoalWrapper(NormaliseEnv(ReduceDCMotorEnv(self._env, reduce_obs=[1])))

    def generate_maze_env(self):
        inp_dict = dict(
            maze_map = self.mazes[self._exp_args['maze_map']],
            continuing_task = bool(self._exp_args['continuing_task']),
        )
        self._maze = self.mazes[self._exp_args['maze_map']]
        if self._exp_args['max_episode_steps'] is not None:
            inp_dict['max_episode_steps'] = self._exp_args['max_episode_steps']
        self._env = gym.make(self._exp_args['env'], **inp_dict)
        randomise_initial_sampling = self._exp_args.get('randomise_initial_sampling', False)
        randomise_initial_sampling_steps = self._exp_args.get('randomise_initial_sampling_steps', 10000)
        self._env = MazeEnvWrapper(self._env, 
                                   reward_type=self._exp_args['reward_type'],
                                   reset_pos_randomisation=randomise_initial_sampling,
                                   reset_randomisation_time=randomise_initial_sampling_steps)


    @classmethod
    def experiment_params(cls):
        super().experiment_params()
        cls._parser.add_argument('--maze-map', type=str, default='single')
        cls._parser.add_argument("--continuing-task", type=int, default=0)
        cls._parser.add_argument("--sparse-scalar", type=float, default=1.)
        cls._parser.add_argument('--randomise-initial-sampling', type=bool, default=False)
        cls._parser.add_argument('--randomise-initial-sampling-steps', type=int, default=10000)
