import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self):
        self.data, self.target = load_iris(return_X_y=True, as_frame=True)
        
        self.data, self.target = self.data.head(10), self.target.head(10)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42, stratify=self.target)
        # print("train la", self.X_train)
        # print("test la", self.X_test)

    def get_data(self):
        return self.data

    def get_label(self):
        return self.target

# dataset = Dataset()
# data = dataset.get_data()
# data['target'] = dataset.get_label()

# # Plotting pairplot using seaborn
# sns.pairplot(data, hue='target', markers=["o", "s", "D"])
# plt.show()