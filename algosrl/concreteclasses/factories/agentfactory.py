from algos import AbstractComponentFactory, get_class_args

from ..wrappedlibs.SB3 import SACSB3Agent,TD3SB3Agent, PPOSB3Agent


class SB3Factory(AbstractComponentFactory):
    _register = []
    def build(self):
        kwargs = get_class_args(self._register[0], self._exp_args)
        kwargs.pop('agent_file_path')
        self._components['agent'] = self._register[0](self._experiment._env, 
                                                      agent_file_path=self._experiment._file_path,
                                                      **kwargs)

class SACFactory(SB3Factory):
    _register = [SACSB3Agent]

class TD3Factory(SB3Factory):
    _register = [TD3SB3Agent]

class PPOFactory(SB3Factory):
    _register = [PPOSB3Agent]