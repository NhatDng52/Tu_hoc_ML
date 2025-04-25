from dataset import Dataset
from nn import MLP
import torch

dataset = Dataset()
data = dataset.get_data()
data['target'] = dataset.get_label()
input = torch.tensor(dataset.X_train.to_numpy())
#print("input la", input)
MLP = MLP(4,[4,4,2,1])
predict =[MLP(i) for i in input]
print("predict la", predict)
label = [torch.tensor(x) for x in dataset.y_train.to_numpy()]
print("label la", label)

loss = sum((i - j) ** 2 for i,j in zip(predict,label))
loss 
# ở đây nó k tạo ra đối tượng biến thông thường , mà sẽ tạo ra đối tượng tensor , thư viện đã overwrite phép cộng nhân mũ ..., đối tượng được tạo ra cũng có các phương thức như backward, grad, ... để tính toán gradient descent
loss.backward()
for i, weight in enumerate(MLP.layers[0].neurons[0].w):
    print(f"grad for weight {i}:", weight.grad)
print(MLP.parameters())

for p in MLP.parameters():
    if p.grad is not None:
        p.data -= 0.01 * p.grad.data
        p.grad = None