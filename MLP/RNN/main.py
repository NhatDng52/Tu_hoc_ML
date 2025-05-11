
import sys
import os
# Thêm thư mục chứa 'util.py' vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import tensor
import torch.nn.functional as F
from nn import RNN
from utils import create_char_level_tokenizer
from loss import CrossEntropyWithSoftmaxLoss
from optimizer import SGD







# đọc file fox_and_grapes.txt
with open("fox_and_grapes.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of text:", len(text))
print("First 100 characters:\n", text[:100])

tokenizer = create_char_level_tokenizer(text)
print("Vocabulary size:", len(tokenizer))
input_seq = tokenizer.fit(text)
print("First 100 tokenized characters:\n", input_seq[:100])

"""
Bây giờ ta đã có văn bản dc mã hóa, nhưng đầu vào RNN là 3D tensor (batch_size, seq_len, feature_size):
    batch_size: kích thước của batch tức số mẫu, này tự quyết định
    seq_len: chiều dài của chuỗi đầu vào, này tự quyết định
    feature_size: là 1 vì mỗi ký tự được chuyển thành 1 số , nếu dùng embedding sau này thì có thể sẽ lớn hơn
"""

"""
    Có nhiều cách tạo dữ liệu, sau đây là cách đc sử dụng 
        chọn seq -> batch_size = tổng số ký tự - seq_len ( ký tự cuối cùng chỉ tính vào nhãn)
        bắt đầu từ seq đầu tiên, dịch qua 1 ký tự ta được seq và input tiếp theo

"""

seq_len = 10
batch_size = len(input_seq) - seq_len 
# Tạo dữ liệu đầu vào và đầu ra
X = []
y = []
for i in range(batch_size):
    X.append(input_seq[i:i + seq_len])
    y.append(input_seq[i + seq_len])
print("X shape:", len(X), len(X[0]))
print("y shape:", len(y))
print("X:", X[:5])
print("y:", y[:5])
inputs= tensor(X[0:20]).unsqueeze(-1) # Đàu vào RNN là 3D nhưng x chỉ có 2D vì feature size = 1 nên bị cắt mất, ta thêm 1 chiều nữa
print("input :", inputs.shape)
labels = torch.stack([F.one_hot(tensor(y[i]), num_classes=len(tokenizer)) for i in range(20)])
print("labels shape:", labels.shape)
# print("labels:", labels)
model = RNN(1, hidden=10,nouts= len(tokenizer),activation_func='tanh')
loss_fn = CrossEntropyWithSoftmaxLoss()
optimizer = SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    # print("outputs  :", outputs)
    # print("prediction shape:", outputs.shape)
    last_chars = outputs[:, -1, :]  # Lấy đầu ra của ký tự cuối cùng
    # print("last chars  :", last_chars)
    loss = loss_fn(last_chars, labels)
    # print("loss:", loss)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Bắt đầu thử sinh chuỗi

num_chars_to_generate = 100

start = "One hot summer’s day,"



for i in range(num_chars_to_generate):
    # Chuyển đổi chuỗi bắt đầu thành tensor
    input_seq = tokenizer.fit(start)
    input_tensor = tensor(input_seq).unsqueeze(0).unsqueeze(-1)  # Thêm batch_size và feature_size
    # Dự đoán ký tự tiếp theo
    with torch.no_grad():
        output = model(input_tensor)
        last_char = output[:, -1, :]  # Lấy đầu ra của ký tự cuối cùng
        predicted_char = torch.argmax(last_char, dim=1).item()
        # print("predicted char", predicted_char)
        start += tokenizer.inverse([predicted_char])  # Chuyển đổi chỉ số thành ký tự
print("story \n",start)