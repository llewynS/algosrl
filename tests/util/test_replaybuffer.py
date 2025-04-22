import pytest
import numpy as np

from algosrl import ReplayBuffer


# TODO: Need to either add the replay buffer or cache results in TestEnv and Simple agent. 
# As it is an integration pulling in the replay buffer is possible. 

class TestIntegrationEnvAgent:
    @classmethod
    def setup_class(cls):
        """ 
        Set-up prior to all tests.
        """

    @classmethod
    def teardown_class(cls):
        """ 
        Teardown after all tests
        """

    def setup_method(self):
        """
        Setup prior to each test
        """
        self.replay = ReplayBuffer(100) 

    def teardown_method(self):
        """
        Teardown after each test
        """
        del self.replay

    def test_replay_a(self):
        experience = {"a":1,
                      "b":2.0,
                      "c":3.5
                        }
        self.replay.collect_experience(experience)
        assert self.replay.maxlen == 100
        assert len(self.replay)==1
        assert self.replay["a"][0]==1
        assert self.replay["b"][0]==2.0
        assert self.replay["c"][0]==3.5
        for i in range(101):
            experience["a"] += 1
            experience["b"] *= 2
            experience["c"] += 0.5
            self.replay.collect_experience(experience)
            if i<99:
                assert len(self.replay)==i+2
            else:
                assert len(self.replay)==100
        assert self.replay["a"][0]==1+100
        assert self.replay["b"][0]==2.0*2**100
        assert self.replay["c"][0]==3.5+50
        assert self.replay._head_index == 1

