from abc import ABC, abstractmethod
from unit import Neuron

class Layer(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def parameters(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def forward(self,x):
        pass
##---------------------------BEGIN LAYER-------------------------------------------------

class FC_Layer(Layer):
    def __init__(self,nin,nout,activation_func = None):
        self.neurons = [Neuron(nin,activation_func) for _ in range(nout)]
        self.activation_func = activation_func
    def __call__(self,x):
            return self.forward(x)
    def parameters(self):
            return [param for neuron in self.neurons for param in neuron.parameters()]
    """đã có call rồi nhưng vẫn làm thêm foward, vì 2 cái có 2 cách gọi khác nhau, và foward thường để người dùng override"""
    def forward(self, x):   
        out = [neuron(x) for neuron in self.neurons]
        return out
    def __str__(self):
        return f"FC_Layer with {len(self.neurons)} neurons"
class ReLU(Layer):
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        return None
    def __str__(self):
        return "RELU"
    def forward(self,x):
        if isinstance(x, list):
            return [xi.clip(min=0) for xi in x]
        else:
            return x.clip(min=0)
class Sigmoid(Layer):
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        return None
    def __str__(self):
        return "Sigmoid"
    def forward(self,x):
        if isinstance(x, list):
            return [1 / (1 + (-xi).exp()) for xi in x]
        else:
            return 1 / (1 + (-x).exp())
class Tanh(Layer):
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        return None
    def __str__(self):
        return "Tanh"
    def forward(self,x):
        if isinstance(x, list):
            return [(xi.exp() - (-xi).exp()) / (xi.exp() + (-xi).exp()) for xi in x]
        else:
            return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
class Softmax(Layer):
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        return None
    def __str__(self):
        return "Softmax"
    def forward(self,x):
        if isinstance(x, list):
            exp_x = [xi.exp() for xi in x]
            return [xi / sum(exp_x) for xi in exp_x]
        else:   
            exp_x = x.exp()
            return exp_x / exp_x.sum(dim=0, keepdim=True)  # Normalize to get probabilities

class Argmax(Layer):
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        return None
    def __str__(self):
        return "Argmax"
    def forward(self,x):
        if isinstance(x, list):
            return [xi.argmax() for xi in x]
        else:
            return x.argmax(dim=0)  # Assuming x is a tensor
##---------------------------END LAYER-------------------------------------------------
