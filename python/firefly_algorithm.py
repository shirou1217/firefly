import numpy as np
import random
import math
import matplotlib.pyplot as plt

class FA:
    def __init__(self, dimen, population, max_iter):
        self.D = dimen             # Dimension of problems
        self.N = population        # Population size
        self.it = max_iter         # Max iteration
        self.Ub = 3*np.ones(dimen) # Upper bound
        self.Lb = 0*np.ones(dimen) # Lower bound
        self.A = 0.97              # Strength
        self.B = 1                 # Attractiveness constant
        self.G = 0.0001            # Absorption coefficient
        
    def fun(self, pop):
        X = np.array(pop)
        funsum = 0
        for i in range(self.D):
            x = X[:,i]
            funsum += x**2 - 10*np.cos(2*np.pi*x)
        funsum += 10*self.D
        return list(funsum)

def main():
    fa = FA(100,100,100)
    pop=np.zeros((fa.N,fa.D))
    for i in range(fa.N):
        for j in range(fa.D):
            pop[i,j] = np.random.uniform(-100,100)       
    fitness = fa.fun(pop)
    
    best_list = []
    best_para_list = []
    best_list.append(min(fitness))
    arr = fitness.index(min(fitness))
    best_para = pop[arr]
    best_para_list.append(best_para)
    
    r_distance = 0
    it = 1
    while it < fa.it:
        for i in range(fa.N):
            for j in range(fa.D):
                steps = fa.A*(random.uniform(0,1)-0.5)*abs(fa.Ub[0]-fa.Lb[0])
                for k in range(fa.N):
                    if fitness[i] > fitness[k]:
                        r_distance += (pop[i][j] - pop[k][j])**2
                        Beta = fa.B*math.e**(-(fa.G*r_distance))
                        xnew = pop[i][j] + Beta*(pop[k][j] - pop[i][j]) + steps
                        if xnew > fa.Ub[0]:
                            xnew = fa.Ub[0]
                        elif xnew < fa.Lb[0]:
                            xnew = fa.Lb[0]
                        pop[i][j] = xnew
                        fitness = fa.fun(pop)
                        best_ = min(fitness)
                        arr_ = fitness.index(best_)
                        best_para_ = pop[arr_]
                        
        best_list.append(best_)
        best_para_list.append(best_para_)
        it+=1
    
    plt.figure(figsize=(15,8))
    plt.xlabel("GENERATIONS", fontsize=20)
    plt.ylabel("VALUE", fontsize=20)
    plt.plot(best_list,linewidth=2,label = "Best value so far ",color="g")
    plt.savefig('best_value_plot.png')


if __name__ == '__main__':
    main()