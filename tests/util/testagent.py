from algosrl import AbstractPolicy, AbstractAgent

class SimplePolicy(AbstractPolicy):
    def __init__(self, action_table):
        self._action_table = action_table
        self._index = -1

    def __call__(self, _):
        if self._index >= (self._action_table.shape[0] - 1):
            self._index = -1
        self._index += 1
        return self._action_table[self._index, :]

    @classmethod
    def set_up_hyperparameters(cls):
        pass


class SimpleAgent(AbstractAgent):
    def __init__(self, policy: AbstractPolicy = None, *args,
                 **kwargs):
        super().__init__(policy=policy, *args, **kwargs)
        
    def initialise(self):
        pass
    
    def step(self):
        # 0
        self.action = self.policy(self.observation)

    def update(self):
        pass
    
    @classmethod
    def set_up_hyperparameters(cls):
        pass

    def load(self):
        pass

    def save(self):
        pass