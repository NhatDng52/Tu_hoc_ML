from abc import ABC, abstractmethod
from torch import tensor
from torch import stack  # dùng stack thay vì cho nó vào tensor mới sẽ ngăn việc grad bị mất
import random
from utils import create_activation_function
from layer import FC_Layer
##---------------------------BEGIN ABSTRACT CLASS-------------------------------------------------


class Model(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def parameters(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

##---------------------------END ABSTRACT CLASS-------------------------------------------------


##---------------------------BEGIN MODEL-------------------------------------------------

class MLP():
    def __init__(self,nin,nouts, activation_func = None,output_activation_func = None):
        #nin will be number of that pass to this MLP , nouts is LIST of hidenlayers and end with output
        size = [nin] + nouts
        self.layers = []
        activation_func = create_activation_function(activation_func)
        end_activation_func = create_activation_function(output_activation_func)                    
            
        for i in range(len(nouts)-1):
            self.layers.append(FC_Layer(size[i],size[i+1]))
        if activation_func:
            self.layers.append(activation_func)

        self.layers.append(FC_Layer(size[-2],size[-1]))
        if end_activation_func:
            self.layers.append(end_activation_func)

            
    def __call__(self,x):
        def process_1_input(x):    
            for layer in self.layers:
                x = layer(x)
            if isinstance(x, list):
                return stack([xi for xi in x])  # Dùng torch.stack để kết hợp tensor và duy trì gradient
            return x  
        if isinstance(x,list):
            return stack([process_1_input(i) for i in x])
        else:
            return process_1_input(x)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters() if layer.parameters() else []
        return params
    
    def __str__(self):
        Layer_list = [str(layer) for layer in self.layers]
        return "MLP with layers:\n" + "\n".join(Layer_list)
    
# class RNN():
#     def __init__(self,nin,hidden,nouts, activation_func = None,output_layer = None,hidden_size = 10):
#         activation_func = create_activation_function(activation_func)
        
#         self.hidden_layer = FC_Layer(hidden,hidden,activation_func)
#         self.output_layer = FC_Layer(hidden,nouts,output_layer)
#         self.input_layer = FC_Layer(nin,hidden,activation_func)
        
#         # cần một biến lưu bộ nhớ 
#         self.ht = None
#     def __call__(self,x):
#     def parameter(self):
#         return [self.weigh_xh,self.weigh_hh,self.weigh_hy]
#     def __str__(self):
#         return f"RNN with weigh_hh \n{self.weigh_hh}\n and weigh_xh \n{self.weigh_xh}\n and weigh_hy \n{self.weigh_hy}\n"
    
    