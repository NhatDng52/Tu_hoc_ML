from dataset import Dataset
from naive_bayes import NaiveBayes
import numpy as np
class Main():
    def __init__(self):
        self.num_of_samples =int( input("chọn sô lượng mẫu để train, số phải bé hơn 5169: "))
        self.dataset = Dataset(self.num_of_samples)
        self.data = self.dataset.get_data()
        index_count = self.data.shape[1]
        self.atributes_index = np.arange(1,index_count)
        self.label_index = np.array(0)
        self.classifier = NaiveBayes(self.data,self.label_index,self.atributes_index)
    def run(self):
        """ 
            Vì không có thời gian nên chưa thể cho nhập text bất kì , mà chỉ cho nhập text có sẵn trong dataset 
        """
        while True:
            text_idx =int( input(f"chọn text index, số phải bé hơn {self.num_of_samples}: "))
            print(f"bạn chọn text : \n\n {self.dataset.data['text'][text_idx-1]}\n\n")
            option = input(" Nếu bạn đồng ý dùng text này để dự đoán thì bấm 1 \n Muốn chọn lại thì bấm 0 \n Các phím khác sẽ là thoát \n Nhập lựa chọn của bạn: ")
            if(option == '1'):
                print(" text bạn vừa chọn được cho rằng là ",self.classifier.predict(self.data[text_idx],self.atributes_index))
            elif(option == "0"):
                continue
            else:
                break
         
        
if __name__ == '__main__':
    main = Main()
    main.run()
