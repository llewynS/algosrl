import gymnasium as gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger
from typing import Dict
import pathlib
import numpy as np
from datetime import datetime
from torch import cuda

from algos.logger.logger import DatabaseLogger
from ....abstractbaseclasses import AbstractAgent, AbstractRLEvaluator, AbstractRLComponent
from algos import Observer, AbstractComponent
from ....util import remove_truncated_if_needed


device = 'cuda' if cuda.is_available() else 'cpu'

class SB3Agent(AbstractAgent):
    default_io_map = {**AbstractAgent.default_io_map.copy(), **{'info': 'info'}}
    # models = {'SAC': SAC, 'TD3': TD3}
    _off_policy_klass = None

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
                 replay_buffer_size:int = 100000,
                 gamma:float = 0.99,
                 warmup:int =10000, 
                 mlp_policy_type:str="MultiInputPolicy",
                 additional_agent_kwargs:Dict[str, any] = {},
                 *args, 
                 **kwargs):
        remove_truncated_if_needed(self, env, io_map)
        if self._off_policy_klass is None:
            raise NotImplementedError("AbstractError: cannot instantiate this class")
        self._evaluator = evaluator
        self.agent_file_path = str(agent_file_path / f"{self._off_policy_klass.__name__}")
        # Define the directory where logs should be saved
        log_dir = self.agent_file_path

        # # Create the TensorBoard writer
        # tensorboard_writer = logger.TensorBoardOutputFormat(log_dir)

        # # Instantiate the logger with the TensorBoard writer
        # custom_logger = logger.Logger(folder=log_dir, output_formats=[tensorboard_writer])
        self._env = env
        self._batch_size = batch_size
        replay_buffer_klass = DictReplayBuffer if mlp_policy_type=="MultiInputPolicy" else ReplayBuffer
        model_kwargs = {"replay_buffer_class": replay_buffer_klass,
                        "verbose": 1, 
                        "learning_rate": learning_rate,
                        "policy_kwargs": dict(net_arch=net_arch),
                        "gamma": gamma,
                        "device": device,
                        **additional_agent_kwargs}
        self.add_to_model_kwargs(model_kwargs)
        self.model = self._off_policy_klass(mlp_policy_type, env, **model_kwargs)
        
        replay_buffer = replay_buffer_klass(buffer_size=replay_buffer_size, observation_space=env.observation_space, 
                        action_space=env.action_space, device=self.model.device)
        self.model.learn(0)
        if model_kwargs.get("use_sde", False):
            self.model.policy.actor.action_dist.exploration_mat = self.model.policy.actor.action_dist.exploration_mat.to(device)
        self.truncated=False
        self.model.replay_buffer = replay_buffer
        self._step = 0
        self._return = 0
        self._epoch = Observer(io_map['epoch'],self)
        self._training_frequency = training_frequency
        self._log_frequency = log_frequency
        self._warmup = warmup
        super().__init__(self.model, io_map=io_map, *args, **kwargs)
        self.model._logger = self._logger_inst

    def initialise(self):
        self.model.logger.record("ep_rew", self._return, ignore_frequency=True)
        self._return = 0

    def step(self):
        obs = self.observation
        self.model.policy.set_training_mode(False)
        if self._step < self._warmup:
            action = np.array([self.model.action_space.sample()])
        else:
            try:
                action, _ = self.model.predict(obs, deterministic=False)
            except Exception as e:
                print(f'observation_shape: {obs["observation"].shape}')
                print(obs)
                raise e
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
        if not AbstractComponent._is_training:
            return
        try:
            self.model.replay_buffer.add(observation, observation_prime, action, reward, done, info)
        except ValueError as e:
            print(f'observation_shape: {observation["observation"].shape}')
            print(f'observation_prime_shape: {observation_prime["observation"].shape}')
            print(f'action shape: {action.shape}')
            print(f'info: {info}')
            print(f'done: {done}')
            print(f'reward: {reward}')
            print(f'actions: {action[:20,:]}')
            raise 
        if self._step > self._warmup:
            if self._step % self._training_frequency == 0:
                retries = 0
                while True:
                    try:
                        self.model.train(gradient_steps=1, batch_size=self._batch_size)
                        break
                    except Exception as e:
                        if retries > 10:
                            raise e
                        retries += 1
                        # print(f"Retrying training: {retries}")
    
    def add_to_model_kwargs(self, kwargs):
        pass

    def load(self, fp = None):
        if fp is None:
            fp = pathlib.Path(f"{self.agent_file_path}") / "agent.zip"
        self.model = self.model.load(str(fp), self._env)

    def save(self):
        self.model.save(f"{self.agent_file_path}/agent")

    @classmethod
    def set_up_hyperparameters(cls):
        pass

    @property
    def epoch(self):
        return self._epoch.state
    
    def evaluate(self):
        if self._evaluator is not None:
            if self._evaluator.will_evaluate(self._step):
                self.model.save(f"{self.agent_file_path}/agent_{self._step}")
            # self._evaluator.evaluate(self._epoch._state if AbstractRLComponent.max_epochs else self._step)

class SACSB3Agent(SB3Agent):
    _off_policy_klass = SAC

    def __init__(self, env: gym.Env, io_map: Dict[str, str] = SB3Agent.default_io_map, training_frequency = 1,
                    batch_size=256, net_arch=[400,300], learning_rate=7e-4,
                    use_sde:bool = True, sde_sample_freq=4, ent_coef=0.0,
                    agent_file_path: pathlib.Path = pathlib.Path(), log_frequency=10,
                    warmup=10000, gamma=0.98, replay_buffer_size=30000, 
                    mlp_policy_type:str="MultiInputPolicy",
                    *args, **kwargs):
        additional_agent_kwargs = dict(
            sde_sample_freq = sde_sample_freq,
            use_sde = use_sde,
        )
        kwargs.pop('additional_agent_kwargs', None)
        super().__init__(env, io_map=io_map, 
                         training_frequency=training_frequency, 
                         batch_size=batch_size,net_arch=net_arch,
                         learning_rate=learning_rate,
                         agent_file_path=agent_file_path,
                         log_frequency=log_frequency,
                         warmup=warmup,
                         gamma=gamma,
                         replay_buffer_size=replay_buffer_size, 
                         mlp_policy_type=mlp_policy_type,
                         additional_agent_kwargs=additional_agent_kwargs,
                         *args, **kwargs)
    @classmethod
    def set_up_hyperparameters(cls):
        cls.hyperparameters['net_arch'].bounds = (100, 800)
        cls.hyperparameters['net_arch'].length = [3, 4]
        cls.hyperparameters['learning_rate'].bounds = (4e-6, 1e-3)
        cls.hyperparameters['batch_size'].bounds = (700, 1000)
        cls.hyperparameters['training_frequency'].bounds = (1, 16)
        cls.hyperparameters['replay_buffer_size'].bounds = (10000, 50000)

class TD3SB3Agent(SB3Agent):
    _off_policy_klass = TD3
    def __init__(self, env: gym.Env, io_map: Dict[str, str] = SB3Agent.default_io_map, training_frequency = 2,
                    batch_size=100, target_policy_noise = 0.1, net_arch=[400,300], learning_rate=1e-3,
                    agent_file_path: pathlib.Path = pathlib.Path(), log_frequency=10, gamma = 0.98,
                    replay_buffer_size = 200000,
                    warmup=10000, 
                    mlp_policy_type:str="MultiInputPolicy",
                    *args, **kwargs):
        self._target_policy_noise = target_policy_noise
        super().__init__(env, io_map=io_map, 
                         training_frequency=training_frequency, 
                         batch_size=batch_size,net_arch=net_arch,
                         learning_rate=learning_rate,
                         agent_file_path=agent_file_path,
                         log_frequency=log_frequency,
                         warmup=warmup,
                         gamma=gamma,
                         replay_buffer_size=replay_buffer_size,
                         mlp_policy_type=mlp_policy_type,
                         *args, **kwargs)
        
    def add_to_model_kwargs(self, kwargs):
        kwargs['target_policy_noise'] = self._target_policy_noise