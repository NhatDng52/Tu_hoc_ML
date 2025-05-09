from abc import ABC, abstractmethod
from torch import tensor

class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self, y_pred, y_true):
        pass
    
    @abstractmethod
    def parameters(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
##---------------------------BEGIN LOSS-------------------------------------------------
class CrossEntropyLoss(Loss):
    def __call__(self, y_pred, y_true):
        # print("y_pred la", y_pred)
        # print("y_true la", y_true)
        if isinstance(y_pred, list)  :
            y_pred = tensor(y_pred)
        if isinstance(y_true, list)  :
            y_true = tensor(y_true)
        loss = -(y_true * y_pred.log()).sum()
        return loss

    def backward(self, y_pred, y_true):
        pass

    def parameters(self):
        return []

    def __str__(self):
        return "CrossEntropyLoss"
class MSELoss(Loss):
    def __call__(self, y_pred, y_true):
        return sum((y_t - y_p) ** 2 for y_t,y_p in zip(y_true,y_pred))

    def backward(self, y_pred, y_true):
        pass

    def parameters(self):
        return []

    def __str__(self):
        return "MSELoss"
##---------------------------END LOSS-------------------------------------------------
