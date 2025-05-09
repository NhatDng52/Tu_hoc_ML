from layer import Layer, FC_Layer, Tanh, ReLU, Sigmoid, Softmax
def create_activation_function(activation_func):
    if activation_func == 'tanh':
        return Tanh()
    elif activation_func == 'relu':
        return ReLU()
    elif activation_func == 'sigmoid':
        return Sigmoid()
    elif activation_func == 'softmax':
        return Softmax()
    else:
        return None