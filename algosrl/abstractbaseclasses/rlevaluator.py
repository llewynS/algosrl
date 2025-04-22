from abc import abstractmethod

from algos import AbstractParametered

class AbstractRLEvaluator(AbstractParametered):
    def __init__(self, frequency: int, *args, **kwargs):
        self._frequency = frequency
        super().__init__(*args, **kwargs)

    @abstractmethod
    def evaluate(self, epoch: int, *args, **kwargs):
        pass
