""" đây là decision tree kết hợp kĩ thuật bagging """
import numpy as np
from dataset import Dataset
from decision_tree import DecisionTree


class RandomForest() :
    def __init__(self,data,label_index,atributes_index,forest_size = 10):
        self.forest = []
        for _ in range (forest_size):
            bagging_dataset = data[np.random.choice(data.shape[0],size=data.shape[0],replace=True)]
            tree_atributes_index=atributes_index
            if atributes_index.shape[0] > 1:
                num_of_atribute = np.random.randint(0,atributes_index.shape[0]-1)
                tree_atributes_index = np.random.choice(atributes_index,size=num_of_atribute,replace=False)
            self.forest.append(DecisionTree(bagging_dataset,label_index,tree_atributes_index))
    def classify(self,data):
        predict_result = []
        for tree in self.forest:
            predict_result.append(tree.classify(data))
        unique, counts = np.unique(predict_result, return_counts=True)
        most_common_label = unique[np.argmax(counts)]
        return most_common_label
            
# #tc1
# data = np.array([
#     [1, 'a'],
#     [2, 'b'],
#     [2, 'a'],
#     [3, 'a'],
#     [4, 'c'],
# ])

# a_index = np.array([])
# print(a_index.size)
# tree = RandomForest(data,label_index=1,atributes_index=a_index)

#tc 2
# print("tc4\n")
# data = np.array([
#     [1,'a',0],
#     [2, 'b',1],
#     [2, 'a',0],
#     [3, 'a',0],
#     [4, 'c',0],
# ])

# a_index = np.array([0,2])
# tree = RandomForest(data,label_index=1,atributes_index=a_index)
# unknow_data = np.array([2,'not know',0])
# print(tree.classify(unknow_data))

#tc3
dataset = Dataset()
label_index = len(dataset.get_data().feature_names)-1
atribute_index = np.arange(0, label_index)
RandomForest = RandomForest(dataset.get_data().data,label_index,atribute_index)