import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger
from typing import Dict
import pathlib
import numpy as np
from datetime import datetime

from algos.logger.logger import DatabaseLogger
from ....abstractbaseclasses import AbstractAgent, AbstractRLEvaluator, AbstractRLComponent
from algos import Observer
from ....util import remove_truncated_if_needed
from .agent import device

class SB3OnPolicyAgent(AbstractAgent):
    default_io_map = {**AbstractAgent.default_io_map.copy(), **{'info': 'info'}}
    # models = {'SAC': SAC, 'TD3': TD3}
    _on_policy_klass = None

    def __init__(self, 
                 env: gym.Env, 
                 evaluator: AbstractRLEvaluator = None, 
                 io_map: Dict[str, str] = default_io_map, 
                 training_frequency = 2,
                 batch_size=100, 
                 net_arch=[256,256], 
                 learning_rate=1e-3,
                 agent_file_path: pathlib.Path = pathlib.Path(), 
                 log_frequency=10, 
                 gamma:float = 0.99,
                 warmup:int =10000, 
                 mlp_policy_type:str="MultiInputPolicy",
                 additional_agent_kwargs:Dict[str, any] = {},
                 *args, 
                 **kwargs):
        remove_truncated_if_needed(self, env, io_map)
        if self._on_policy_klass is None:
            raise NotImplementedError("AbstractError: cannot instantiate this class")
        self._evaluator = evaluator
        self.agent_file_path = str(agent_file_path / f"{self._on_policy_klass.__name__}")
        # Define the directory where logs should be saved
        # log_dir = self.agent_file_path

        # # Create the TensorBoard writer
        # tensorboard_writer = logger.TensorBoardOutputFormat(log_dir)

        # # Instantiate the logger with the TensorBoard writer
        # custom_logger = logger.Logger(folder=log_dir, output_formats=[tensorboard_writer])
        self._env = env
        self._batch_size = batch_size
        model_kwargs = {"verbose": 1, 
                        "learning_rate": learning_rate,
                        "policy_kwargs": dict(net_arch=net_arch),
                        "gamma": gamma,
                        "device": device,
                        **additional_agent_kwargs}
        self.add_to_model_kwargs(model_kwargs)
        self.model = self._on_policy_klass(mlp_policy_type, env, **model_kwargs)
        self.model.learn(0)
        if model_kwargs.get("use_sde", False):
            self.model.policy.action_dist.exploration_mat = self.model.policy.action_dist.exploration_mat.to(device)
        self.truncated=False
        self._step = 0
        self._return = 0
        self._n_steps = 0
        self._epoch = Observer(io_map['epoch'],self)
        self._training_frequency = training_frequency
        self._log_frequency = log_frequency
        self._warmup = warmup
        super().__init__(self.model, io_map=io_map, *args, **kwargs)
        self.model._logger = self._logger_inst

    def initialise(self):
        self.model.logger.record("ep_rew", self._return)
        self._return = 0
        self._n_steps = 0

    def step(self):
        if self.model.use_sde and self.model.sde_sample_freq > 0 and self.n_steps % self.model.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(self._env.num_envs)
        obs = self.observation
        self.model.policy.set_training_mode(False)
        if self._step < self._warmup:
            action = np.array([self.model.action_space.sample()])
        else:
            
            action, _ = self.model.predict(obs, deterministic=False)
            # print(f"ActionSHAPE: {action.shape} at step: {self._step}")
            if action.shape[0]>1:
                # print("action_shape is batch size")
                print(f'observation shape: {obs["observation"].shape}')
                action, _ = self.model.predict(obs, deterministic=False)
                print(f"Action SHAPE after resampling: {action.shape} at step: {self._step}")
        self.action=action
        self._step = self.exp_step
        self.model.logger.set_step(self._step) 

    def update(self):
        
        info = self.info 
        done = self.terminal
        observation = self.observation
        observation_prime = self.observation_prime
        action = self.action
        reward = self.reward
        self.model.logger.record("reward", reward[0])
        self._return += reward[0]
        if self._step > self._warmup:
            if self._step % self._training_frequency == 0:
                self.model.train()
    
    def add_to_model_kwargs(self, kwargs):
        pass

    def load(self):
        self.model.save(f"{self.agent_file_path}/agent", self.env)

    def save(self):
        self.model.save(f"{self.agent_file_path}/agent")

    @classmethod
    def set_up_hyperparameters(cls):
        pass

    @property
    def epoch(self):
        return self._epoch.state
    
    def evaluate(self):
        # print("EVALUATING")
        if self._evaluator is not None:
            self._evaluator.evaluate(self._epoch._state if AbstractRLComponent.max_epochs else self._step)

class PPOSB3Agent(SB3OnPolicyAgent):
    _on_policy_klass = PPO
    def __init__(self, env: gym.Env, io_map: Dict[str, str] = SB3OnPolicyAgent.default_io_map, 
                    gamma = 0.99,
                    vf_coef = 0.5,
                    ent_coef = 0.0,
                    max_grad_norm = 0.5,
                    normalize_advantage = True,
                    n_steps = 512,
                    n_epochs = 20,
                    sde_sample_freq = 4,
                    use_sde = True,
                    gae_lambda = 0.9,
                    clip_range = 0.4,
                    batch_size=128, 
                    net_arch=[256,256], 
                    learning_rate=3e-5,
                    agent_file_path: pathlib.Path = pathlib.Path(),
                    log_frequency=10, 
                    warmup=10000, 
                    mlp_policy_type:str="MultiInputPolicy",
                    *args, **kwargs):
        additional_agent_kwargs = dict(
            vf_coef = vf_coef,
            ent_coef = ent_coef,
            max_grad_norm = max_grad_norm,
            normalize_advantage = normalize_advantage,
            n_steps = n_steps,
            n_epochs = n_epochs,
            sde_sample_freq = sde_sample_freq,
            use_sde = use_sde,
            gae_lambda = gae_lambda,
            clip_range = clip_range
        )
        kwargs.pop('additional_agent_kwargs', None)
        super().__init__(env, io_map=io_map, 
                         batch_size=batch_size,net_arch=net_arch,
                         learning_rate=learning_rate,
                         agent_file_path=agent_file_path,
                         log_frequency=log_frequency,
                         warmup=warmup,
                         gamma=gamma,
                         mlp_policy_type=mlp_policy_type,
                         additional_agent_kwargs=additional_agent_kwargs,
                         *args, **kwargs)
        
