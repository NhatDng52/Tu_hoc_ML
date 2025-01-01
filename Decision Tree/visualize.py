import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from dataset import Dataset
class Visualize():
    def __init__(self):
        print(" visualize create")
    def visualize_data(self,data):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['label'] = data.target
        print(df)
    def visualize_tree(self,decision_tree):   
        decision_tree.print_tree()

