import numpy as np
import math

def entropy(data, label_index):
    total_samples =len(data)
    unique_values,count = np.unique(data[:,label_index],return_counts=True)
    entropy = 0.0 
    probabilities = count / total_samples
    for prob in probabilities :
        entropy +=  -prob * math.log(prob,2)
    return entropy


def info_gain(data, atribute_index , label_index):
    total_samples =len(data)
    gain = 0.0 + entropy(data,label_index)
    unique_values,count = np.unique(data[:,atribute_index],return_counts=True)
    subtables = {value: data[data[:, atribute_index] == value] for value in unique_values}
    probabilities = count / total_samples
    for value, subtable in subtables.items():
        gain -= probabilities[list(unique_values).index(value)] * entropy(subtable, label_index)
    return gain
    
    
    
    
    
# data = np.array([
#     [1, 'a'],
#     [2, 'b'],
#     [2, 'a'],
#     [3, 'a'],
#     [4, 'c'],
#     [1, 'c'],
#     [2, 'b'],
#     [2, 'a'],
#     [3, 'b'],
#     [4, 'd'],
# ])
# print(data.shape)
# print(entropy(data, 0))  # Output: 4
# print(entropy(data, 1))  # Output: 3
# print(info_gain(data,0,1))