from typing import Sequence

from .module import Parameter
from .scalar import Scalar

import numpy as np

class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                """
                print('-'*40)
                print('p type:', type(p))
                print('p:', p)
                print('p.value.derivative:', p.value.derivative)
                print('history:', p.value.history)
                """
                #p.value.derivative += np.random.randint(-100, 100)
                #if p.value.unique_id % 100 == 0:
                    #print('p:', p)
                    #print('deriv:', p.value.derivative)
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
