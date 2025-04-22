from torch import ones as torch_ones
from torch import tensor as torch_tensor
from torch import stack as torch_stack
from torch import float32 as torch_float32
from torch.distributions import MixtureSameFamily, Uniform, Independent, Categorical

from . import device

import numpy as np
from typing import List

def get_empty_cells(maze) -> List[np.ndarray]:
    """Returns all the empty cells in the maze in discrete cell coordinates (i,j)."""
    empty_cells = []
    for i in range(maze.map_length):
        for j in range(maze.map_width):
            if maze.maze_map[i][j] == 0:
                x = (j + 0.5) * maze.maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze.maze_size_scaling
                empty_cells.append(np.array([x, y]))
    return empty_cells
#Pretty sure I want this distribution to be in xy coordinates and not row col coordinates
def generate_target_distribution_maze(env, use_empty_cells=False, use_start_cells=False):
    # Identify 'g' squares
    if hasattr(env, 'envs'):
        env = env.envs[0]
    g_coords = env.maze.unique_goal_locations[:]
    if use_empty_cells:
        g_coords += get_empty_cells(env.maze)
    if use_start_cells:
        g_coords += env.maze.unique_reset_locations[:]
    cell_size = env.maze.maze_size_scaling
    offset = cell_size/2.0
    distributions = [Uniform(low=torch_tensor([x-offset, y-offset], dtype=torch_float32).to(device), 
                             high=torch_tensor([x+offset, y+offset], dtype=torch_float32).to(device)) for x, y in g_coords]

    # Concatenate distributions using a MixtureSameFamily distribution
    # Assuming equal mixture weights for simplicity
    weights = torch_ones(len(distributions), dtype=torch_float32).to(device) / len(distributions)
    low = torch_stack([dist.low for dist in distributions]).to(device)
    high = torch_stack([dist.high for dist in distributions]).to(device)
    mixture_distribution = MixtureSameFamily(
        mixture_distribution=Categorical(weights),
        component_distribution=Independent(
            Uniform(low=low,
                    high=high),
            reinterpreted_batch_ndims=1
        )
    )

    return mixture_distribution