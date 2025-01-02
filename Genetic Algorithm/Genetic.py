""" Giải thuật di truyền được lấy cảm hứng từ thuyết tiến hóa trong sinh học
    Sử dụng các ý tưởng như hoán vị, tổ hợp, đột biến
    Giải thuật được sử dụng ở các bài toán tổ hợp ( như Knapshack), khi mà các không gian search quá lớn
    Ở bài toán Knapsack, khi mà threshold không biết trước, chúng ta sẽ có thể:
        Xem xét sự tăng giảm của fitness trong quần thể để quyết định dừng
        Giới hạn số lượng đời con ( số lần lặp )
"""
import numpy as np
from dataset import Dataset
from function import fitness
class GeneticAlgorithm:
    def __init__(self,data,capacity,fitness,limit = 10,p=100,r=0.5,m=0.5):
        self.data = data
        self.limit = limit
        self.r = r
        self.m = m
        self.p = p
        self.fitness = fitness
        self.capacity = capacity
        self.children = []
        self.population = []
        self.fitness_values=[]
        self.prev_fitness_max = float('-inf')
        self.cur_fitness_max =float('-inf')
        self.prev_max_hypothesis = np.array([])
    def GA(self,init_prob = 0.05):
        decreased_time = 3 # sau 3 lượt không có thế hệ mới có fit lớn hơn, dừng thuật toán
        for _ in range(self.p):
            hypothesis = np.array([0 for i in range(self.data.shape[0])])
            random_values = np.random.rand(hypothesis.shape[0])
            # print(random_values)
            hypothesis[random_values < init_prob] = 1
            self.population.append(hypothesis)
        #print(self.population[0].shape)
        
        for hyp in self.population :
            self.fitness_values.append(self.fitness(self.data,hyp,self.capacity))
        self.cur_fitness_max = max(self.fitness_values)
        while self.cur_fitness_max < float('inf') :
            self.prev_fitness_max = self.cur_fitness_max
            sum_fitness = np.sum(self.fitness_values)
                
            self.fitness_values = np.array(self.fitness_values) / sum_fitness
            chosen_indices = np.random.choice(len(self.population), int( (1-self.r) * self.p), replace=False, p=self.fitness_values)
            chosen = [self.population[i] for i in chosen_indices]
            
            #chọn các cá thể từ đời bố mẹ cho crossover , ở đây mình dùng tournament selection 
            
            num_of_tournaments = (self.p - int( (1-self.r) * self.p))*2
            winners = [] 
            for _   in range(num_of_tournaments):
                competitors_indices = np.random.choice(len(self.population), 2, replace=False, p=self.fitness_values)
                competitors = [self.population[i] for i in competitors_indices]
                # print("competitor ",competitors)
                competitors_fitness = [self.fitness_values[i] for i in competitors_indices]
        

                winner_index = np.argmax(competitors_fitness)
                #print('winner index is',winner_index," and it shape is", competitors[winner_index].shape)
                winners.append(competitors[winner_index])
            winners = np.array(winners)
            # print(self.fitness_values)
            #print("winers shape ",winners.shape)
            crossover_children = []
            for i in range(winners.shape[0]//2):
                # việc chọn giải thuật crossover cũng là một vấn đề cần quan tâm , ở đây đơn giản nhất mình sẽ chọn 50% cha 50% mẹ đan xen
                crossover_mask = np.array([i%2 for i in range(self.data.shape[0])])
                child = winners[i*2] * crossover_mask + winners[i*2+1] * np.logical_not(crossover_mask).astype(int)
                crossover_children.append(child)
            #print(len(crossover_children))
            self.children = np.concatenate((chosen, crossover_children), axis=0)
            
            
            # Mutation 
            mutation_children_indices =  np.random.choice(len(self.children), int(self.m * len(self.children)), replace=False)
            mutation_children = [self.children[i] for i in mutation_children_indices]
            #print('mutation children', int(self.m * len(self.children)),"\n\n")
            for mutation_child in mutation_children:
                index = np.random.randint(0,len(mutation_child))
                np.logical_not(crossover_mask).astype(int)
                mutation_child[index]= np.logical_not(mutation_child[index]).astype(int)
            for i,j in zip(mutation_children_indices,mutation_children):
                self.children[i] = j
            self.population = self.children
            self.fitness_values = []
            for child in self.children:
                self.fitness_values.append(self.fitness(self.data,child,self.capacity))
            self.cur_fitness_max = max(self.fitness_values)
            
            if(self.cur_fitness_max < self.prev_fitness_max):
                decreased_time-=1
                if(decreased_time == -1):
                    break
            if(self.limit == 0):
                break
            else :
                self.limit -= 1
        print(f"fitness value is {max(self.fitness_values)}")
        return self.fitness_values   
            
ds = Dataset()
# (fitness(ds.get_data(4),3000000))
capa = ds.get_label(4)['capacity']      
test = GeneticAlgorithm(ds.get_data(4),capa,fitness,100)

test.GA()
