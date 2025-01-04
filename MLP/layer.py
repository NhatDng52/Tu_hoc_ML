import numpy as np
import function as f
class FullyConnected:
    def __init__(self,in_dim,out_dim,use_bias = True ):
        self.weight = np.zeros(in_dim,out_dim)
        self.use_bias = use_bias
        self.cached_X = 0
        if(use_bias):
            self.bias = np.zeros(out_dim)
            
        """" Backward data """   
         
        self.grad_weight = 0
        self.grad_bias = 0
          
    def forward(self, X):
        self.cached_X = X
        result = np.dot(X,self.weight)
        if (self.use_bias):
            result += self.bias
        return result

    def backward(self, DY):
        if(self.use_bias):
            self.grad_bias = DY
        self.grad_weight = np.dot(np.transpose(self.cached_X),DY)
        return self.grad_weight
    def get_param(self):
        if(self.use_bias):
            return self.weight,self.bias
        else :
            return self.weight
    def get_grad(self):
        if(self.use_bias):
            return self.grad_weight,self.grad_bias
        else :
            return self.grad_weight       
class Tanh:
    def __init__(self):
        self.cached_X = 0
        
        """" Backward data """   
        self.grad_value = 0
    def forward(self, X):
        self.cached_X = X
        return np.tanh(X)
    def backward(self, DY):
        return (1 - np.square(np.tanh(self.cached_X))) * DY
    def get_param(self):
        return 0
    def get_grad(self):
        return 0      
class ReLU:
    def __init__(self):
        self.mask = None
        
        """" Backward data """   
        self.grad_value = 0
    def forward(self, X):
        self.mask = np.where(X>0)
        return X * self.mask
    def backward(self, DY):
        return DY * self.mask
    def get_param(self):
        return 0
    def get_grad(self):
        return 0 

class Softmax:
    def __init__(self):
        self.cache_predict = 0
        """" Backward data """   
        self.grad_value = None
    def foward(self,input_list):
        result = f.matrix_softmax(input_list)
        self.cache_predict = result
        return result
    def backward(self,DY,y_true):
        for i in range(self.cache_predict.shape[0]):
            label = y_true[i]
            for j in range(self.cache_predict.shape[1]):
                
                if( j == label):
                    self.grad_value[i][j] = self.cache_predict[i][j] * ( 1 - self.cache_predict[i][j])
                else :
                    self.grad_value[i][j] = - self.cache_predict[i][j] * self.cache_predict[i][label]
        return self.grad_value * DY
    def get_param(self):
        return 0
    def get_grad(self):
        return 0 
        
class CrossEntropy:
    def __init__(self):
        self.loss = None
        self.grad_value = None
        self.y_predict = None
        self.y_true = None
    def calculate_loss(self,y_predict,y_true):
        self.y_true = y_true
        self.y_predict = y_predict
        for i in range(y_true.shape[0]):
            label = y_true[i]
            self.loss[i] = -np.log(y_predict[label]) 
    def backward(self):
        for i in range(self.y_predict.shape[0]):
            label = self.y_true[i]
            self.grad_value[i] = -1/(self.y_predict[label] +1e-10)