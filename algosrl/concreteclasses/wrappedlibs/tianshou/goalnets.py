import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import gymnasium as gym
from tianshou.utils.net.common import MLP

def create_linear_relu(obs_dim, initial_layer_size):
    #Check len obs_dim and add flatten layer if greater than 1
    if len(obs_dim) == 1:
        layer = nn.Sequential(
                nn.Linear(obs_dim[0], initial_layer_size),
                nn.ReLU()
            )
    else:
        layer = nn.Sequential(
                nn.Flatten(start_dim=0),
                nn.Linear(np.prod(obs_dim), initial_layer_size),
                nn.ReLU()
            )
    return layer

class DictEnvNet(nn.Module):
    def __init__(self, env: gym.Env, initial_layer_size = 64, concat_layer_size = 128, hidden_layers = [64, 64], activation=nn.ReLU, device: Union[str, int, torch.device] = "cpu"):
        super().__init__()
        self.device = device
        self._inputs = {}
        #check if env.observation_space is Dict or OrderedDict
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            for key in env.observation_space.keys():
                obs_dim = env.observation_space[key].shape
                self._inputs[key] = create_linear_relu(obs_dim, initial_layer_size)


            model = []
            # Combine the outputs of the observation components
            combined_dim = initial_layer_size * len(self._inputs)  # Output sizes of individual components
            model += [
                nn.Linear(combined_dim, concat_layer_size),
                nn.ReLU()
            ]
            # Hidden layers
            for i, layer_size in enumerate(hidden_layers):
                if i == 0:
                    model += [
                        nn.Linear(concat_layer_size, layer_size),
                        nn.ReLU()
                    ]
                else:
                    model += [
                        nn.Linear(hidden_layers[i-1], layer_size),
                        nn.ReLU()
                    ]
            # Output layer
            if len(env.action_space.shape) == 1:
                model += [nn.Linear(128, env.action_space.shape[0])]
            else:
                model += [
                    nn.Linear(128, np.prod(env.action_space.shape)),
                    nn.Unflatten(1, env.action_space.shape)
                ]
            self.output_dim = np.prod(env.action_space.shape)
            self.model = nn.Sequential(*model)

    def forward(self, obs_dict, state=None, info={}, act=None):
        outputs = [self._inputs[key](obs_dict[key]) for key in self._inputs.keys()]
        if act:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            outputs.append(act)
        combined_output = torch.cat(outputs, dim=1)
        return self.model(combined_output), state


class DictCritic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        logits, hidden = self.preprocess(obs, act)
        logits = self.last(logits)
        return logits

