from torch import tensor
import random


# Ở đây ta sẽ làm việc với tensor thay vì numpy , tensor là đơn vị tính toán cơ bản của DL, nó hỗ trợ GPU , nguyên nhân khiến Deep Learning bùng nổ lại ở những năm gần đây, ngoài ra nó còn support các chức năng thêm (như grad, thứ k thể thiếu của back prog )
class Neuron():
    def __init__(self,nin,activation_func = None):
        self.w = [tensor(random.uniform(-1,1),requires_grad= True) for _ in range(nin)]
        self.b = tensor(random.uniform(-1,1),requires_grad= True)
        self.activation_func = activation_func
    def __call__(self,x):
        # x is list of inputs , and we have list of weights
        out =  sum((w*x_i for w,x_i in zip(self.w,x)) , self.b)
        out = self.activation_func(out) if self.activation_func else out
        return out
    def parameters(self):
        return self.w+ [self.b]

class Layer():
    def __init__(self,nin,nout,activation_func = None):
        self.neurons = [Neuron(nin,activation_func) for _ in range(nout)]
        self.activation_func = activation_func
    def __call__(self,x):
        out = [neuron(x) for neuron in self.neurons]
        return out
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
class MLP():
    def __init__(self,nin,nouts,activation_func = None):
        #nin will be number of that pass to this MLP , nouts is LIST of hidenlayers and end with output
        size = [nin] + nouts
        self.layers = [Layer(size[i],size[i+1],activation_func) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]