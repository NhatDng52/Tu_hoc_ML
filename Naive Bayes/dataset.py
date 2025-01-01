""" 
        Thuật toán này khá đơn giản ,theo mình thấy khó ở chỗ chuẩn bị dataset 
    khi bài toán điển hình của thuật toán này là việc lọc spam,
    nhưng dataset thực tế thì chứa các dòng text độ dài khác nhau, không dễ để biến nó thành bảng để có được xác suất riêng rẽ
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
class Dataset():
    def __init__(self,num_of_samples = 0):
        # Load the dataset
        file_path = './spam.csv'  # Replace with the actual path to your spam.csv file
        self.data = pd.read_csv(file_path, encoding='latin-1')

        # Keep only the columns 'v1' and 'v2'
        self.data = self.data[['v1', 'v2']]
        
        # Rename columns for clarity
        self.data.columns = ['label', 'text']
        
        # Slice the first 100 rows
        self.table = self.data.head(1)
        self.truncate(num_of_samples)
    
    def truncate(self,row):
        self.table = self.data.head(row)
        self.vectorizer = CountVectorizer(stop_words='english')
        temp = self.vectorizer.fit_transform(self.table['text']).toarray()
        self.table = self.table.drop(['text'],axis =1)
        self.table = self.table.to_numpy()
        self.table = np.concatenate((self.table,temp),axis = 1)
        self.feature_names = self.vectorizer.get_feature_names_out()
    def get_data(self):
        return self.table
    
    def transfrom_to_bow(self,text):
        return self.vectorizer.transform([text]).toarray()

