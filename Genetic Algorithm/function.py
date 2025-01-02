import numpy as np
from dataset import Dataset
def fitness(table,hypothesis,capa):
    data = table[hypothesis == 1]
    result = np.sum(data,axis=0)
    #print(f"result 1 is {result[1]}")
    if(result[1] < capa):
        return result[0]
    else: 
        return 1e-10
    
    
# ds = Dataset()

# print(fitness(ds.get_data(1),3000000))