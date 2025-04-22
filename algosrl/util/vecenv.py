from stable_baselines3.common.vec_env import DummyVecEnv
from copy import deepcopy
import numpy as np

class MyDummyVecEnv(DummyVecEnv):
    """MyDummyVecEnv prevents the standard behavvior of DummyVecEnv to reset the environment when the episode is done.
    This is to prevent resets within the training loop, outside of AlgOSRL Control.
    """
    def step_wait(self):
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            # if not isinstance(self.actions, list):
            #     self.actions = [self.actions]
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))