import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from algos import ComponentCollection, Subject
from algosrl import StandardRLEnv, AbstractRLComponent
from algosrl.concreteclasses.wrappedlibs.SB3.agent import SACSB3Agent
from algosrl.defaultnamespace import *

class TestIntegrationEnvAgent:
    @classmethod
    def setup_class(cls):
        """ 
        Set-up prior to all tests.
        """
        pass
    @classmethod
    def teardown_class(cls):
        """ 
        Teardown after all tests
        """
        pass

    def setup_method(self):
        """
        Setup prior to each test
        """
        AbstractRLComponent.max_epochs = 10
        env = gym.make('parking-v0')
        env = DummyVecEnv([lambda: env])

        envRL = StandardRLEnv(env)
        agent = SACSB3Agent(env)
        Subject.verify_obs_subs()
        self._cc = ComponentCollection(**{
            'agent': agent,
            'env': envRL,
        }) 
        # for ac in self._cc.keys:
        #     self._cc[ac].set_component_collection(self._cc)

    def teardown_method(self):
        """
        Teardown after each test
        """
        del self._cc

    def test_integration_agent_env(self):
        self._cc.run()
        
