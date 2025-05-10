from abc import ABC, abstractmethod
from torch import tensor,zeros,ones
import random

##---------------------------BEGIN UNIT-------------------------------------------------

# Ở đây ta sẽ làm việc với tensor thay vì numpy , tensor là đơn vị tính toán cơ bản của DL, nó hỗ trợ GPU , nguyên nhân khiến Deep Learning bùng nổ lại ở những năm gần đây, ngoài ra nó còn support các chức năng thêm (như grad, thứ k thể thiếu của back prog )
class Neuron():
    def __init__(self,nin,activation_func = None):
        """
            Khi khởi tạo :
            - Không nên để tất cả trọng số = 0
            - Không nên để tất cả trọng số giống nhau ( như 1 hết hoặc 2 hết)
            - Không nên để trọng số quá lớn hoặc quá nhỏ
            Cách ok nhất là ở trong khoảng -1 đến 1 , 
            có thể tham khảo bài viết :  https://medium.com/data-science/neural-network-breaking-the-symmetry-e04f963395dd      
        """
        self.w = [tensor(random.uniform(-1,1), requires_grad=True) for _ in range(nin)]
        self.b = tensor(random.uniform(-1,1), requires_grad=True)
        self.activation_func = activation_func
    def __call__(self,x):
        # x is list of inputs , and we have list of weights
        out =  sum((w*x_i for w,x_i in zip(self.w,x)) , self.b)
        out = self.activation_func(out) if self.activation_func else out
        return out
    def parameters(self):
        return self.w+ [self.b]

##---------------------------END UNIT-------------------------------------------------
