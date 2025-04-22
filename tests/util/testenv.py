import numpy as np
from gymnasium import spaces

class DummyEnv:
    def __init__(self):
        self.observation_space = spaces.Box(-10, 10, shape=(3, 1))
        self.action_space = spaces.Box(-10, 10, shape=(3, 1))
        self._state = None
        self._can_die = None

    def step(self, action):
        self._state = action * 2.0
        if self._can_die is not None:
            terminal = np.random.choice([0.0, 1.0], size=1, p=[0.8, 0.2])
        else:
            terminal = 0.0
        return self.state, 0.0, False ,terminal, {}

    def reset(self):
        self._state = np.array([0.1, 0.1, 0.1]).reshape(
            (3, 1))    #np.random.randint(-10, 10, size=(3, 1))
        return self.state, {}

    @property
    def state(self):
        return self._state
