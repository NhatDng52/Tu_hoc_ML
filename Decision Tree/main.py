""" Decision tree thuộc bài toán phân loại, nó có thể phân loại dữ liệu thành các nhãn ( các ví dụ cơ bản thường phân nó với nhãn YES/NO)"""
import numpy as np
from dataset import Dataset
from decision_tree import DecisionTree
from visualize import Visualize
class Main():
    def __init__(self):
        self.dataset = Dataset()
        self.visualize = Visualize()
        label_index = len(self.dataset.get_data().feature_names)-1
        atribute_index = np.arange(0, label_index)
        self.decision_tree_ID3 = DecisionTree(self.dataset.get_data().data,label_index,atribute_index)
    def run(self):
        self.visualize.visualize_data(self.dataset.data())
        print(f"toal node = {self.decision_tree_ID3.call}")
        # self.visualize.visualize_tree(self.decision_tree_ID3)
      
     
if __name__ == '__main__':
    main = Main()
    main.run()
