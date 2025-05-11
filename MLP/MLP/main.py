
import sys
import os
# Thêm thư mục chứa 'util.py' vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import Dataset
from nn import*
import torch
import torch.nn.functional as F
dataset = Dataset()
data = dataset.get_data()
from loss import MSELoss, CrossEntropyLoss, CrossEntropyWithSoftmaxLoss
from optimizer import SGD

#--------------------------------------------- Iris dataset------------------------------
data['target'] = dataset.get_label()
inputs = torch.tensor([x for x in dataset.X_train.to_numpy()])
labels = [torch.tensor(x) for x in dataset.y_train.to_numpy()]
# print("label la", labels)
labels = torch.stack([F.one_hot(label.long(), num_classes=3) for label in labels])
#----------------------------------------------XOR dataset-----------------------------------------  
# Tạo inputs cho XOR dataset (4 mẫu)
# inputs =torch.tensor ([
#     [1.0, 1.0],
#     [1.0, 0.0],
#     [0.0, 1.0],
#     [0.0, 0.0]
# ])

# labels = torch.tensor([
#     [1,0],  # [0, 0] -> 0
#     [0,1],  # [0, 1] -> 1
#     [0,1],  # [1, 0] -> 1
#     [1,0]   # [1, 1] -> 0
# ], dtype=torch.int32)


#------END XOR dataset
print("inputs la", inputs)
print("labels la", labels)

model = MLP(4,[4,2,3], activation_func='relu')
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyWithSoftmaxLoss()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(inputs)
    # print("output la", output)
    loss = loss_fn(output, labels)
    loss.backward()
    # for param in model.parameters():
    #     print(param.grad)
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#Prediction
with torch.no_grad():
    data['target'] = dataset.get_label()
    inputs = torch.tensor([x for x in dataset.X_test.to_numpy()])
    labels = torch.tensor([torch.tensor(x) for x in dataset.y_test.to_numpy()])
    # print("label la", labels)
    # labels = torch.stack([F.one_hot(label.long(), num_classes=3) for label in labels])
    output = model(inputs)
    predicted_labels = torch.argmax(output, dim=1)
    print("predicted_labels la", predicted_labels)
    print("labels la", labels)
    acc = (predicted_labels == labels).float().mean()
    print(f"Accuracy: {acc.item() * 100:.2f}%")
