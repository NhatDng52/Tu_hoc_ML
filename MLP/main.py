from dataset import Dataset
from nn import*
import torch
import torch.nn.functional as F
dataset = Dataset()
data = dataset.get_data()
from loss import MSELoss
from optimizer import SGD

#--------------------------------------------- Iris dataset------------------------------
# data['target'] = dataset.get_label()
# inputs = [torch.tensor(x) for x in dataset.X_train.to_numpy()]
# labels = [torch.tensor(x) for x in dataset.y_train.to_numpy()]
# # print("label la", labels)
# labels = torch.stack([F.one_hot(label.long(), num_classes=3) for label in labels])
#----------------------------------------------XOR dataset-----------------------------------------  
# Tạo inputs cho XOR dataset (4 mẫu)
inputs = [
    torch.tensor([1.0, 1.0], dtype=torch.float64),
    torch.tensor([1.0, 0.0], dtype=torch.float64),
    torch.tensor([0.0, 1.0], dtype=torch.float64),
    torch.tensor([0.0, 0.0], dtype=torch.float64)
]

labels = torch.tensor([
    [0],  # [0, 0] -> 0
    [1],  # [0, 1] -> 1
    [1],  # [1, 0] -> 1
    [0]   # [1, 1] -> 0
], dtype=torch.int32)


#------END XOR dataset
model = MLP(2,[2,1], activation_func='relu', output_activation_func=None)
optimizer = SGD(model.parameters(), lr=0.1)
loss_fn = MSELoss()
for epoch in range(100):
    preds = model(inputs)
    loss = loss_fn(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("pred", preds)
    print("loss", loss)

























# # test dự đoán 
# test_input = [torch.tensor(x) for x in dataset.X_test.to_numpy()]
# test_labels = torch.stack([torch.tensor(x) for x in dataset.y_test.to_numpy()])

# # test XOR
# test_input = inputs
# test_labels = labels

# output = model(test_input)


# print("label", test_labels)

# # # Tìm chỉ số lớp với xác suất cao nhất trong output
# _, predicted_classes = torch.max(output, dim=1)
# print("pred",predicted_classes)
# # # Tính độ chính xác bằng cách so sánh predicted_classes và test_labels
# # correct = (predicted_classes == test_labels).sum().item()
# # accuracy = correct / test_labels.size(0)

# # print(f"Accuracy: {accuracy * 100:.2f}%")