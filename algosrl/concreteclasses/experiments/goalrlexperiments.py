from .rlmazeexperiment import SB3RLExperiment
from ..factories.agentfactory import SACFactory, TD3Factory
from ..factories.envfactory import EnvFactory
from ..factories.goalfactory import UniformFactory
from ..factories.evaluatorfactory import GoalEvaluatorComponentFactory

class SACUniformExperiment(SB3RLExperiment):
    _builders = [UniformFactory, SACFactory, EnvFactory, GoalEvaluatorComponentFactory]

class TD3UniformExperiment(SB3RLExperiment):
    _builders = [UniformFactory, TD3Factory, EnvFactory, GoalEvaluatorComponentFactory]