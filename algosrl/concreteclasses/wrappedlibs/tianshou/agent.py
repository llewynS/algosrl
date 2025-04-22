import torch, numpy as np, torch.nn as nn
from typing import Dict, Any, Union, List
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.policy import SACPolicy, TD3Policy, DDPGPolicy
from tianshou.data import ReplayBuffer

from ....abstractbaseclasses import AbstractAgent


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OffPolicyTianshouAgent(AbstractAgent):
    default_io_map = {**AbstractAgent.default_io_map.copy(), **{'info': 'info'}}

    def __init__(self, policy, io_map: Dict[str, str] = default_io_map, batch_size=64, *args, **kwargs):
        super().__init__(policy, io_map=io_map, *args, **kwargs)
        self.replay = ReplayBuffer(size=100000)
        self._step = self._steps_this_epoch = 0
        self._logger = ts.utils.TensorboardLogger(SummaryWriter(f'log/{self.__class__.__name__}'))
        self.batch_size = batch_size


    def initialise(self):
        self._steps_this_epoch = 0

    def step(self):
        self.action = self.policy(Batch(obs = self.observation.reshape(-1,*self._state_shape), info=self.info)).act.data.numpy().reshape(*self._action_shape)
        self._step += 1

    def update(self):
        _ = self.epoch
        batch_dict = {"obs": self.observation.reshape(-1,*self._state_shape), 
                      "act": self.action.reshape(-1,*self._action_shape), 
                      "rew" : self.reward,
                      "obs_next": self.observation_prime.reshape(-1,*self._state_shape),
                      "terminated": self.terminal,
                      "truncated": self.truncated,
                      "info": self.info}
        self.replay.add(Batch(**batch_dict))
        losses = self.policy.update(self.batch_size, self.replay)                                    
        self._logger.log_update_data(losses, self._step)

    def load(self):
        pass

    def save(self):
        pass

    @classmethod
    def set_up_hyperparameters(cls):
        pass

class SAC(OffPolicyTianshouAgent):
    def __init__(self, state_shape: np.array, action_shape: np.array, replay: ReplayBuffer=None,
                 io_map: Dict[str, str] = OffPolicyTianshouAgent.default_io_map,
                 hidden_sizes: List[int]=[256,256], actor_lr: float=1e-3, critic_lr: float=1e-3,
                 alpha_lr: float=3e-4, alpha: float=0.2, auto_alpha: bool=True, 
                 tau: float=0.005, gamma: float=0.99, n_step:int =1, *args, **kwargs):
        self._state_shape = state_shape
        self._action_shape = action_shape
        net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = ActorProb(
            net_a,
            action_shape,
            device=device,
            unbounded=True,
            conditioned_sigma=True,
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device,
        )
        net_c2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device,
        )
        critic1 = Critic(net_c1, device=device).to(device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=device).to(device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        if auto_alpha:
            target_entropy = -np.prod(action_shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)

        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            estimation_step=n_step,
            action_space=action_shape,
            )
        super().__init__(policy, io_map=io_map, *args, **kwargs)

class TD3(OffPolicyTianshouAgent):
    def __init__(self, state_shape: np.array, action_shape: np.array, max_action: np.array, 
                 replay: ReplayBuffer, io_map: Dict[str, str] = OffPolicyTianshouAgent.default_io_map,
                 hidden_sizes: List[int]=[256,256], actor_lr: float=3e-4, critic_lr: float=3e-4,
                 tau: float=0.005, gamma: float=0.99, exploration_noise: float=0.1, policy_noise: float=0.2,
                 update_actor_freq: int=2, noise_clip: float=0.5, n_step:int =1, *args, **kwargs):
        exploration_noise = exploration_noise * max_action
        policy_noise = policy_noise * max_action
        noise_clip = noise_clip * max_action
        net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = Actor(
            net_a, action_shape, max_action=max_action, device=device
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device,
        )
        net_c2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device,
        )
        critic1 = Critic(net_c1, device=device).to(device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=device).to(device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        policy = TD3Policy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            estimation_step=n_step,
            action_space=action_shape,
        )
        super().__init__(policy, replay, io_map, *args, **kwargs)

class DDPG(OffPolicyTianshouAgent):
    def __init__(self, state_shape: np.array, action_shape: np.array, max_action: np.array,
                 replay: ReplayBuffer, io_map: Dict[str, str] = OffPolicyTianshouAgent.default_io_map,
                 hidden_sizes: List[int]=[256,256], actor_lr: float=1e-3, critic_lr: float=1e-3,
                 tau: float=0.005, gamma: float=0.99, exploration_noise: float=0.1, n_step:int=1, *args, **kwargs):
        exploration_noise =  exploration_noise *  max_action
        # model
        net_a = Net( state_shape, hidden_sizes= hidden_sizes, device= device)
        actor = Actor(
            net_a,  action_shape, max_action= max_action, device= device
        ).to( device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr= actor_lr)
        net_c = Net(
             state_shape,
             action_shape,
            hidden_sizes= hidden_sizes,
            concat=True,
            device= device,
        )
        critic = Critic(net_c, device= device).to( device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr= critic_lr)
        policy = DDPGPolicy(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau= tau,
            gamma= gamma,
            exploration_noise=GaussianNoise(sigma= exploration_noise),
            estimation_step= n_step,
            action_space=action_shape,
        )
        super().__init__(policy, replay, io_map, *args, **kwargs)