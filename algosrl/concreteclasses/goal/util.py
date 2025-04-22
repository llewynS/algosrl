import gymnasium as gym
import numpy as np
from typing import List

from highway_env.vehicle.objects import Landmark



def get_maze_cells(maze, cell_type = [0]) -> List[np.ndarray]:
    """Returns all the empty cells in the maze in discrete cell coordinates (i,j)."""
    empty_cells = []
    for i in range(maze.map_length):
        for j in range(maze.map_width):
            if maze.maze_map[i][j] in cell_type:
                x = (j + 0.5) * maze.maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze.maze_size_scaling
                empty_cells.append(np.array([x, y]))
    return empty_cells

#The below is cooked and should not be used because it's doing min max of row_col and not the x,y coordinates
def get_goal_bounds_maze_env(env: gym.Env):
    valid_locations = get_maze_cells(env.maze, cell_type=[0, "g"])
    return np.min(valid_locations, axis=0),np.max(valid_locations, axis=0)

def get_goal_bounds_parking_env(env: gym.Env):
    array = np.vstack([lane.position(lane.length/2, 0) for lane in env.road.network.lanes_list()])
    return np.min(array, axis = 0), np.max(array, axis=0)

def get_goal_bounds(env: gym.Env):
    if hasattr(env, 'envs') and not env.envs[0].spec.id == 'DCMotor-v0':
        env = env.envs[0]
    if hasattr(env, 'maze'):
        return get_goal_bounds_maze_env(env)
    elif hasattr(env, 'road'):
        return get_goal_bounds_parking_env(env)
    else:
        return env.observation_space['desired_goal'].low, env.observation_space['desired_goal'].high

def transform_goal_maze_env(goal):
    return goal

def transform_goal_parking_env(env, goal):
    heading = 1.5707963267948966
    if goal[1] < 0:
        heading = -1.5707963267948966
    return Landmark(env.road, goal, heading)

def transform_goal(env: gym.Env, goal):
    if hasattr(env, 'envs'):
        env = env.envs[0]
    if hasattr(env, 'maze'):
        return transform_goal_maze_env(goal)
    elif hasattr(env, 'road'):
        return transform_goal_parking_env(env, goal)
    elif env.spec.id == 'DCMotor-v0':
        return goal
    else:
        raise NotImplementedError(f"Goal transform not implemented for {env}")
    
