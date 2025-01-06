from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
class Dataset():
    def __init__(self):
        """ vì wine load ở dạng object nên cần trích xuất data, đổi tên và gán label để bên module visualize làm việc dễ hơn"""
        self.wine =  load_wine()
        df = pd.DataFrame(self.wine.data, columns=self.wine.feature_names)
        self.wine.feature_names.append('label')
        df['label'] = self.wine.target
        self.wine.data = df.to_numpy()
        
 
    def get_data(self):
        return self.wine