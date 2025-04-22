from ..goal.goalevaluator import GoalEvaluator
from ..goal.goalevaluatorcomponent import GoalEvaluatorComponent
from algos import AbstractComponentFactory, get_class_args

class GoalEvaluatorFactory(AbstractComponentFactory):
    _register = [GoalEvaluator]

    def build(self):
        evaluator = self._register[0](self._experiment._file_path,
                                      self._experiment._components['agent']._policy,
                                      self._experiment._env,
                                      self._exp_args['max_steps'],
                                      **get_class_args(self._register[0], self._exp_args))
        self._experiment._components['agent']._evaluator = evaluator
        if 'model' in self._experiment._components.keys:
            self._experiment._components['model']._evaluator = evaluator
        if 'goal' in self._experiment._components.keys:
            self._experiment._components['goal']._evaluator = evaluator


class GoalEvaluatorComponentFactory(AbstractComponentFactory):
    _register = [GoalEvaluatorComponent]

    def build(self):
        evaluator = self._register[0](self._experiment._env,
                                      self._exp_args['max_steps'],
                                      **get_class_args(self._register[0], self._exp_args))
        self._experiment._components['agent']._evaluator = evaluator
        if 'model' in self._experiment._components.keys:
            self._experiment._components['model']._evaluator = evaluator
        if 'goal' in self._experiment._components.keys:
            self._experiment._components['goal']._evaluator = evaluator
        if 'env' in self._experiment._components.keys:
            self._experiment._components['env']._evaluator = evaluator
            evaluator.set_rl_env(self._experiment._components['env'])
        self._experiment._components['evaluator'] = evaluator