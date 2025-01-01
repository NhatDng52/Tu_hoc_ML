"""
    Bài toán phân loại sử dụng xác suất có điều kiện là bài toán tìm xác suất P(Y|X)  trong đó Y và X là các vector đặc trưng
    Giả sử X gồm 5 đặc trưng , Y chỉ gồm 1 và mỗi đặc trưng chỉ có 2 giá trị ( kiểu bool )
    Xác suất trên có thể viết dưới dạng :
    
                                            P(y|x1,x2,x3,x4,x5)
                                            
    Vấn đề này thực tế rất khó , khi số lượng tham số theo cấp số mũ ( ví dụ như trên ta sẽ cần 2*(2^5-1) biến để lưu xác suất với 2^5-1 là giá trị khác nhau của vector X và 2 là 2 giá trị của Y   )
    Theo ví dụ của tác giả Tom Mitchell trong sách machine learning , rằng chỉ cần vector X có 30 đặc trưng thôi, chúng ta phải cần lưu đến gần 3 tỉ tham số
    Vì thể cần những giả sử để bài toán này có thể áp dụng được, Naive Bayes là một trong số đó
    
    Naive Bayes giả sử rằng các biến xi độc lập với xj khác xi, và độc lập với tập X-{xi}  Nếu biết Y  ( khái niệm này gọi là độc lập có điều kiện)
    
    Lúc này sử dụng định lý Bayes : P(y|x1,x2,x3,x4,x5) = P(x1,x2,x3,x4,x5|y)  *P(y) / P(x1,x2,x3,x4,x5)  =  P(x1,x2,x3,x4,x5|y)  *P(y) / [xích ma(P(yj) * P(x1,x2,x3,x4,x5|yj)) ] 
    Khi có giả thuyết độc lập có đk thì xác suất của các vector có thể tách thành tích xác suất các đặc trưng riêng lẻ, khiến tham số của X phải quản lí xuống còn 2n  (3n nếu 3 giá trị ..)
    
    Vậy thì tại sao lại không giả sử  Y độc lập có đk với các X-{xi} khác, khi đã biết xi ? Nếu thế thì tính luôn không cần biến đổi Bayes ?
        - Vì mục tiêu ban đầu của bài toán phân loại là Y phụ thuộc vào nhiều thuộc tính của X , nếu làm như trên nó chỉ cần dùng 1x, quá vô lí với thực tế  
     
"""
from dataset import Dataset
import numpy as np
class NaiveBayes():
    def __init__(self,data,label_index,atributes_index):
        print (f"init called")
        self.prob_table = {}
        self.create_probability_table(data,label_index,atributes_index)
       
    def create_probability_table(self,data,label_index,atributes_index):
        unique_labels = np.unique(data[:,label_index]) # xem data có bao nhiêu nhãn
        for label in unique_labels:
            label_table = data[data[:, label_index] == label][:, atributes_index]
            num_of_samples = label_table.shape[0]
            print(f"label: {label},has num {num_of_samples} ")
            self.prob_table[label] = np.sum(label_table,axis =1)
            self.prob_table[label] /= num_of_samples
        return self.prob_table
        


dataset = Dataset()
data = dataset.get_data()
index_count = data.shape[1]
atributes_index = np.arange(1,index_count)
label_index = np.array(0)
classifier = NaiveBayes(data,label_index,atributes_index)