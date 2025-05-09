from abc import ABC, abstractmethod
from torch import tensor

class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
##---------------------------BEGIN OPTIMIZER-------------------------------------------------
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.data.zero_()

    def __str__(self):
        return f"SGD with learning rate {self.lr}"
# ##---------------------------END OPTIMIZER-------------------------------------------------