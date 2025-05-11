from abc import ABC, abstractmethod
from torch import tensor,is_tensor,zeros,ones
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
    def __init__(self,nin,nouts, activation_func = None):
        #nin will be number of that pass to this MLP , nouts is LIST of hidenlayers and end with output
        size = [nin] + nouts
        self.layers = []
        activation_func = create_activation_function(activation_func)                   
            
        for i in range(len(nouts)):
            self.layers.append(FC_Layer(size[i],size[i+1]))
        if activation_func and i != len(nouts)-1:
            self.layers.append(activation_func)
    def __call__(self,x):
        if not is_tensor(x):
            raise ValueError("Input must be a tensor. If input is a list, please wrap it in a tensor.")
        def process_1_input(x):   
            for layer in self.layers:
                x = layer(x)
            return x  

        return stack([process_1_input(x[i]) for i in range(x.shape[0])])
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters() if layer.parameters() else []
        return params
    
    def __str__(self):
        Layer_list = [str(layer) for layer in self.layers]
        return "MLP with layers:\n" + "\n".join(Layer_list)
    
class RNN():
    def __init__(self,nin,hidden,nouts, activation_func = None):
        activation_func = create_activation_function(activation_func)
        
        self.output_layer = FC_Layer(hidden,nouts)
        self.input_layer = FC_Layer(nin,hidden)
        
        #hidden layer là fc và activation_func nếu có
        self.hidden_layer = []
        self.hidden_layer.append(FC_Layer(hidden,hidden))
        if activation_func:
            self.hidden_layer.append(activation_func)
            
        self.hidden_state = None 
        self.hidden_size = hidden
    def __call__(self,x):
        if not is_tensor(x):
            raise ValueError("Input must be a tensor. If input is a list, please wrap it in a tensor.")

        def process_1_seq(x): 
            self.hidden_state = zeros(self.hidden_size)

            def process_1_input(x):
                if len(self.hidden_layer) == 1:
                    self.hidden_state = self.input_layer(x) + self.hidden_layer[0](self.hidden_state)
                else:
                    self.hidden_state = self.hidden_layer[1](self.input_layer(x) + self.hidden_layer[0](self.hidden_state))
                out = self.output_layer(self.hidden_state)
                return out
            return stack([process_1_input(x[i]) for i in range(x.shape[0])])
        return stack([process_1_seq(x[i]) for i in range(x.shape[0])]) 
          
        
    def parameters(self):
        params = []
        params += self.input_layer.parameters() if self.input_layer.parameters() else []
        for layer in self.hidden_layer:
            params += layer.parameters() if layer.parameters() else []
        params += self.output_layer.parameters() if self.output_layer.parameters() else []
        return params
            
    def __str__(self):
        return f"RNN with input layer {self.input_layer}, hidden layer {self.hidden_layer}, output layer {self.output_layer}"
    
    