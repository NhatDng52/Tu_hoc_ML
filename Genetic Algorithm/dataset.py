""" 
    Data được lấy từ https://www.kaggle.com/datasets/sc0v1n0/large-scale-01-knapsack-problems
    Data cần phải nhiều item nhiều weight, nếu quá ít genetic algo cũng có thể sử dụng được, nhưng nó thua giải thuật search thông thường   
"""
import pandas as pd

class Dataset():
    def __init__(self):
        case_1 = pd.read_csv('./knapPI_1_500_1000_1_items.csv').to_numpy()
        info_1 = pd.read_csv('./knapPI_1_500_1000_1_info.csv').to_numpy()
        self.data_1 = case_1[:,1:3] 
        self.solution_1 = {} 
        self.solution_1['data'] = case_1[case_1[:,3]==1][:,1:3] 
        self.solution_1['capacity'] = info_1[0][1]
        self.solution_1['value'] = info_1[1][1]
        
        case_2 = pd.read_csv('./knapPI_2_500_1000_1_items.csv').to_numpy()
        info_2 = pd.read_csv('./knapPI_2_500_1000_1_info.csv').to_numpy()
        self.data_2 = case_2[:,1:3] 
        self.solution_2 = {} 
        self.solution_2['data'] = case_2[case_2[:,3]==1][:,1:3] 
        self.solution_2['capacity'] = info_2[0][1]
        self.solution_2['value'] = info_2[1][1]
        
        case_3 = pd.read_csv('./knapPI_3_500_1000_1_items.csv').to_numpy()
        info_3 = pd.read_csv('./knapPI_3_500_1000_1_info.csv').to_numpy()
        self.data_3 = case_3[:,1:3] 
        self.solution_3 = {} 
        self.solution_3['data'] = case_3[case_3[:,3]==1][:,1:3] 
        self.solution_3['capacity'] = info_3[0][1]
        self.solution_3['value'] = info_3[1][1]
        
        case_4 = pd.read_csv('./knapPI_14_200_1000_1_items.csv').to_numpy()
        info_4 = pd.read_csv('./knapPI_14_200_1000_1_info.csv').to_numpy()
        self.data_4 = case_4[:,1:3] 
        self.solution_4 = {} 
        self.solution_4['data'] = case_4[case_4[:,3]==1][:,1:3] 
        self.solution_4['capacity'] = info_4[0][1]
        self.solution_4['value'] = info_4[1][1]
    def get_data(self,case_num):
        if(case_num == 1):
            return self.data_1
        elif(case_num == 2):
            return self.data_2
        elif(case_num == 3):
            return self.data_3
        else:
            return self.data_4        
    def get_label(self,case_num):
        if(case_num == 1):
            return self.solution_1
        elif(case_num == 2):
            return self.solution_2
        elif(case_num == 3):
            return self.solution_3
        else:
            return self.solution_4        


# ds= Dataset()
# while True:
#     a = int(input('nhap so de xuat test :' ))
#     if( a ==0): 
#         break
#     print(ds.get_data(a))
#     print(ds.get_label(a))