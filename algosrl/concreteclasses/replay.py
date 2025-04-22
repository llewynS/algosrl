from typing import Dict

from ..util import ReplayBuffer
from ..abstractbaseclasses import AbstractRLComponent
from  ..defaultnamespace import DEFAULTEXPERIENCE


class ReplayBufferComponent(AbstractRLComponent):
    default_io_map = {x:x for x in DEFAULTEXPERIENCE}
    def __init__(self, maxlen:int=1000000, io_map:Dict[str, str]=default_io_map, *args, **kwargs):
        rp_keys = list(io_map.keys())
        rp_keys.pop(rp_keys.index('wait'))
        self.replay = ReplayBuffer(maxlen, rp_keys)
        super().__init__(io_map, *args, **kwargs)
        # self.step = gate_observers(*inputs)(self.step)

    def initialise(self):
        pass

    def step(self):
        # if self.terminal:
        #     return
        experience = {x:getattr(self, f'{x}') for x in self.replay.keys}
        self.replay.collect_experience(experience)
        self.wait = 1

    def update(self):
        pass

    def done(self):
        self.replay[-1]['terminal'] = True
        super().done()

    @classmethod
    def set_up_hyperparameters(cls):
        pass

    def load(self):
        pass

    def save(self):
        pass


        