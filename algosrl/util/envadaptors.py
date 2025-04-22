import numpy as np

from typing import Optional, Dict

from gymnasium import Wrapper, spaces
# from .wrapper import Wrapper
from ..concreteclasses.goal import UniformSamplerNotComponent
from ..abstractbaseclasses.rlcomponent import AbstractRLComponent

"""
Add time remaining to observation iff time limit exists
"""

"""Need to handle SB3 Dummy vec env calling reset itself on terminate...
"""

class SetGoalWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._sampler = None

    def set_sampler(self, sampler):
        self._sampler = sampler

    def get_sampler(self):
        return self._sampler
    
    # If I have normalised the observations then this will set the goal to a normalised goal
    # this must occur after normalisation and reduction.
    def set_goal(self, obs = None, goal=None):
        if goal is None:
            if self._sampler is None:
                goal = self.observation_space['desired_goal'].sample()
            else:
                if obs is None:
                    raise ValueError("Must provide obs if sampler is set")
                goal = self._sampler.sample(obs['observation'])
        self.env.set_goal(goal)
        obs['desired_goal'] = goal
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.set_goal(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

class ReduceDCMotorEnv(Wrapper):
    def __init__(self, env, reduce_obs):
        super().__init__(env)
        self._reduce_obs = reduce_obs
        self.observation_space = spaces.Dict(self.env.observation_space)
        self.observation_space['observation'] = spaces.Box(low=self.observation_space['observation'].low[self._reduce_obs], 
                                                           high=self.observation_space['observation'].high[self._reduce_obs], 
                                                           shape=(env.observation_space['observation'].shape[0]-len(self._reduce_obs),),
                                                           dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.reduce_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.reduce_obs(obs), reward, terminated, truncated, info
    
    def reduce_obs(self, obs):
        obs_copy = obs.copy()
        obs_copy['observation'] = obs_copy['observation'].copy()[self._reduce_obs]
        return obs_copy

class NormaliseEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(self.env.observation_space)
        self.observation_space['observation'] = spaces.Box(low=-1.0, high=1.0, shape=env.observation_space['observation'].shape, dtype=np.float32)
        self.observation_space['desired_goal'] = spaces.Box(low=-1.0, high=1.0, shape=env.observation_space['desired_goal'].shape, dtype=np.float32)
        self.observation_space['achieved_goal'] = spaces.Box(low=-1.0, high=1.0, shape=env.observation_space['achieved_goal'].shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalise_observation(obs), info

    def set_goal(self, goal):
        self.env.set_goal(self.unnormalise(goal, 
                                               self.env.unwrapped.observation_space['desired_goal'].low, 
                                               self.env.unwrapped.observation_space['desired_goal'].high))

    def step(self, action):
        action = self.unnormalise_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        norm_obs = self.normalise_observation(obs)
        # distance = np.linalg.norm(norm_obs['achieved_goal'] - norm_obs['desired_goal']) < 0.1
        # info['success'] = distance
        # if distance:
        #     reward = distance.astype(np.float32)*self.env.unwrapped._sparse_scalar
        return norm_obs, reward, terminated, truncated, info

    def normalise_observation(self, obs):
        obs_copy = obs.copy()
        self.normalise_dict_obs(obs_copy, 'observation')
        self.normalise_dict_obs(obs_copy, 'desired_goal')
        self.normalise_dict_obs(obs_copy, 'achieved_goal')
        return obs_copy

    def normalise_dict_obs(self, obs, key):
        obs_low = self.env.observation_space[key].low
        obs_high = self.env.observation_space[key].high
        obs[key] = self.normalise(obs[key], obs_low, obs_high)

    def normalise(self, x, x_min, x_max):
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    def unnormalise(self, x, x_min, x_max):
        return ((x + 1.0) / 2.0) * (x_max - x_min) + x_min

    def unnormalise_action(self, action):
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        return self.unnormalise(action, action_low, action_high)

class MazeEnvWrapper(Wrapper):
    def __init__(self, maze_env, reward_type = "dense", continuing_task=True, reset_pos_randomisation=False, reset_randomisation_time = 10000):
        super().__init__(maze_env)
        #Add dimension to observation space for time remaining
        if hasattr(self.env.spec, 'max_episode_steps'):
            obs_shape: tuple = self.observation_space['observation'].shape
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(
                        -np.inf, np.inf, shape=(obs_shape[0] + 1,), dtype="float32"
                    ),
                    achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float32"),
                    desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float32"),
                )
            )
        self._sampler = None
        self.reward_type = reward_type
        self._evaluate = False
        self._use_normal_goal_setting = False
        self._continuing_task = continuing_task
        self._wrapper_goal = None
        self._reset_pos_sampler = None
        self._step = 0
        self._reset_randomisation_time = reset_randomisation_time
        if reset_pos_randomisation:
            self._reset_pos_sampler = UniformSamplerNotComponent(self.env, 1, 1)

    def set_evaluate(self):
        self._evaluate = True

    def unset_evaluate(self):
        self._evaluate = False
        
    def set_sampler(self, sampler):
        self.env.unwrapped.goal = None
        self._sampler = sampler

    def get_sampler(self):
        return self._sampler

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):  
        if self._use_normal_goal_setting:
            reset_pos, info = self.env.reset(seed=seed, options=options)
            self.concat_time_left(reset_pos)
        else:
            reset_pos, info = self.generate_target_goal(seed, options)
        return reset_pos, info


    def step(self, action):
        self._step += 1
        obs_dict, _, terminated, truncated, info = self.env.step(action)
        if hasattr(self.env, 'ant_env'):
            terminated = terminated or not self.env.ant_env.is_healthy
        reward = self.compute_reward(obs_dict['achieved_goal'], self._wrapper_goal, info)
        terminated = terminated or (self._continuing_task and self.reward_type == "sparse" and reward == 100.0)
        self.update_goal(obs_dict['achieved_goal'], obs_dict)
        self.concat_time_left(obs_dict)
        return obs_dict, reward, terminated, truncated, info

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        reached_goal = int(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
        info['success'] = bool(reached_goal)
        if self.reward_type == "dense":
            if hasattr(self.env, 'ant_env'):
                return sum([info[key] for key in info.keys() if "reward" in key]) + 5*reached_goal 
            else:
                return -(1 - np.exp(-np.linalg.norm(desired_goal - achieved_goal))) + 5*reached_goal
        elif self.reward_type == "sparse":
            return 100.0 if reached_goal else 0.0
        
    def linalg_compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        reached_goal = (np.linalg.norm(achieved_goal - desired_goal, axis=1) <= 0.45).astype(np.int32)
        info['success'] = reached_goal.astype(np.bool_)
        if self.reward_type == "dense":
            if hasattr(self.env, 'ant_env'):
                return sum([info[key] for key in info.keys() if "reward" in key]) + 5*reached_goal 
            else:
                return -(1 - np.exp(-np.linalg.norm(desired_goal - achieved_goal, axis=1))) + 5*reached_goal
        elif self.reward_type == "sparse":
            return reached_goal.astype(np.float32)*100.0
    
    def do_env_reset_pos(self, reset_pos, seed=None):
        self.env.unwrapped.point_env.init_qpos[:2] = reset_pos
        obs, info = self.env.unwrapped.point_env.reset(seed=seed)
        obs_dict = self.env.unwrapped._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )
        return obs_dict, info
    
    def random_reset_pos(self):
        self.env.reset()
        reset_pos = self._reset_pos_sampler.sample(None)
        return self.do_env_reset_pos(reset_pos)

    def generate_target_goal(self, seed=None, options=None) -> np.ndarray:
        assert len(self.maze.unique_reset_locations) > 0
        if self._reset_pos_sampler and AbstractRLComponent._is_training and self._step < self._reset_randomisation_time:
            reset_pos, info = self.random_reset_pos()
        else:
            reset_pos, info = self.env.reset(seed=seed, options=options)
        self.concat_time_left(reset_pos)
        if self._sampler is None:
            self._sampler = UniformSamplerNotComponent(self.env)      
        reset_goal = self.add_xy_position_noise(self._sampler.sample(reset_pos['observation']))        
        maze_pos = self.maze.cell_xy_to_rowcol(reset_goal)
        while self.check_maze_pos(maze_pos) or np.linalg.norm(reset_pos['observation'][:2] - reset_goal) <= 0.5 * self.env.unwrapped.maze.maze_size_scaling:
            reset_goal = self.add_xy_position_noise(self._sampler.sample(reset_pos['observation']))
            maze_pos = self.maze.cell_xy_to_rowcol(reset_goal)
        self.env.unwrapped.goal = reset_goal
        self._wrapper_goal = reset_goal
        self.env.update_target_site_pos()
        reset_pos = self.env.get_wrapper_attr('_get_obs')(reset_pos['observation'])
        
        info["success"] = bool(
            np.linalg.norm(reset_pos["achieved_goal"] - self._wrapper_goal) <= 0.45
        )
        return reset_pos, info
    
    def reevaluate_goal(self, maze_pos, reset_pos, reset_goal, previous_goal):
        if self._evaluate:
            return self.check_maze_pos(maze_pos) or np.linalg.norm(reset_goal - reset_pos['observation'][:2]) < 0.5
        else:
            return np.linalg.norm(reset_goal - previous_goal) < 0.5 or self.check_maze_pos(maze_pos) or np.linalg.norm(reset_goal - reset_pos['observation'][:2]) < 0.5

    def check_maze_pos(self, maze_pos):
        try:
            return self.maze.maze_map[maze_pos[0]][maze_pos[1]]==1
        except IndexError:
            return True


    def generate_target_goal_with_current_position(self, current_pos) -> np.ndarray:
        assert len(self.maze.unique_reset_locations) > 0
        previous_goal = self._wrapper_goal
        reset_goal = self.add_xy_position_noise(self._sampler.sample(current_pos))
        maze_pos = self.maze.cell_xy_to_rowcol(reset_goal)
        # While goal position is close to previous goal position or goal is in wall or goal is close to reset position
        while np.linalg.norm(reset_goal - previous_goal) < 0.5 or self.maze.maze_map[maze_pos[0]][maze_pos[1]]==1 or np.linalg.norm(reset_goal - current_pos[:2]) < 0.5:
            print(f'resampling: g2close: {np.linalg.norm(reset_goal - previous_goal) < 0.5} inwall: {self.maze.maze_map[maze_pos[0]][maze_pos[1]]==1} g2close2reset: {np.linalg.norm(reset_goal - current_pos[:2]) < 0.5}')
            reset_goal = self.add_xy_position_noise(self._sampler.sample(current_pos))
            maze_pos = self.maze.cell_xy_to_rowcol(reset_goal)
        return reset_goal
    
    def update_goal(self, achieved_goal: np.ndarray, current_pos:Dict[str, np.ndarray]) -> None:
        """Update goal position if continuing task and within goal radius."""

        if (
            self.continuing_task
            and self.reset_target
            and bool(np.linalg.norm(achieved_goal - self._wrapper_goal) <= 0.45)
            and len(self.maze.unique_goal_locations) > 1
        ):
            print("UPDATING TARGET GOAL IN STEP")
            # Generate a goal while within 0.45 of achieved_goal. The distance check above
            # is not redundant, it avoids calling update_target_site_pos() unless necessary
            while np.linalg.norm(achieved_goal - self._wrapper_goal) <= 0.45:
                # Generate another goal
                reset_goal = self.generate_target_goal_with_current_position(current_pos['observation'])
                # Add noise to goal position
                self.env.unwrapped.goal = self.add_xy_position_noise(reset_goal)
                current_pos['desired_goal'] = self._wrapper_goal

            # Update the position of the target site for visualization
            self.update_target_site_pos()
            
    def concat_time_left(self, obs_dict):
        if hasattr(self.env.spec, "max_episode_steps"):
            obs_dict['observation'] = np.concatenate((obs_dict['observation'], np.array([self.env.spec.max_episode_steps - self.env._elapsed_steps])))