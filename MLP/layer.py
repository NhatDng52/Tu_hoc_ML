from abc import ABC, abstractmethod
from unit import Neuron
from torch import tensor,stack
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
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
            return self.forward(x)
    def parameters(self):
            return [param for neuron in self.neurons for param in neuron.parameters()]
    """đã có call rồi nhưng vẫn làm thêm foward, vì 2 cái có 2 cách gọi khác nhau, và foward thường để người dùng override"""
    def forward(self, x):  
        out = stack([neuron(x) for neuron in self.neurons])
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
            # print("x in ce",x)
            # input sẽ là 1 tensor 2d để truy cập các vec 1d dùng dim 1 shape không bị thu lại từ 4,1 sang 4 ta dùng keepdim , trả về tuple value,index ta dùng [0]
            # Trừ x với xmax để tránh giá trị quá lớn, nó không làm thay đổi xác suất cuối, vì lúc đóc cả 2 tử và mẫu sẽ chia cho exp(xmax)
            x = x - x.max(dim=1, keepdim=True)[0]  
            exp_x = x.exp()
            # print("exp_x in ce",exp_x)
            return exp_x / exp_x.sum(dim=1, keepdim=True) + 1e-8  # Thêm một giá trị nhỏ để tránh chia cho 0  

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
