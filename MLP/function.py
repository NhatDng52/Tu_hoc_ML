import numpy as np
def matrix_softmax(X):
    print(X.shape)
    """ softmax cho ma trận MxN với M là số samples , gọi softmax 1xN  """
    result = np.zeros_like(X)
    if(X.ndim ==1):
        return softmax(X)
    for i in range(X.shape[0]):  
        for j in range(X.shape[1]):  
            result[i, j] = softmax(X[i, :])[j]
    return result

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()