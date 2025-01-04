from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
""""
    Mình sẽ dùng dataset iris từ sklearn 
        data gồm cột 1 2 là sepal(lá cây) length và width , cột 3 4 là petal( cánh hoa) length và with
        target sẽ là 0,1,2 tương ứng với 3 loại Iris Setosa, Iris Versicolor, Iris Virginica
"""

class Dataset():
    def __init__(self):
        self.data,self.target = load_iris( return_X_y=True, as_frame=True,)
        self.data,self.target = self.data.head(10),self.target.head(10)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.data, self.target, test_size=0.2, random_state=42, stratify=self.target)
        print("train la",self.X_train)
        print("test la",self.X_test)
    def get_data(self):
        return self.data.to_numpy()
    def get_label(self):
        return self.taret.to_numpy()
    
    
dataset = Dataset()
print(dataset.get_data())




