# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:08:48 2019

@author: tekin
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def veri_oku(p,X,Y):
    veriler=[]
    with open("E:\DERSLER\Yüksek Lisans\Optimization\gr137.tsp") as dt:
        data = dt.read()
        for d in data.split("NODE_COORD_SECTION")[1:]:
            veriler=(d.splitlines()[1:-1])
    p=[0]
    X=[]
    Y=[]
    for i in veriler:
        p.append(int(i.split(" ")[0]))
        X.append(float(i.split(" ")[1]))
        Y.append(float(i.split(" ")[2]))
    return p[:-1],X,Y


def simulatedAnnealing(p,xmutlak,ymutlak,ITER_MAX):
    N = len(xmutlak)
    def objFunc(p):
        cost = 0
        for i in range(0,N-1):
            xd = xmutlak[p[i]] - xmutlak[p[i+1]]
            yd =ymutlak[p[i]] - ymutlak[p[i+1]]
            dxy = math.sqrt(xd * xd + yd *yd)
            cost = cost + dxy
        return cost
    #iteration=0
    #init_p = [0,1,2,3,4]
    E_best = objFunc(p)
    E_curr = objFunc(p)
    E_new = objFunc(p)
    
    print("First solution:",E_best)
    result = []
    T_start = 100000
    cooling_factor = 0.99
    T = T_start
    p_new = p.copy()
    p_curr = p.copy()
    iter = 0
    for i in range(ITER_MAX):
        while T > 0.001:
            swap1 = math.floor(random.uniform(0,1) * N)
            swap2 = math.floor(random.uniform(0,1) * N)
            while swap1 == swap2:
                swap2 = math.floor(random.uniform(0,1) * N)
            p_new = p_curr.copy()
            temp = p_new[swap1]
            p_new[swap1] = p_new[swap2]
            p_new[swap2] = temp
            E_new = objFunc(p_new)
            Delta_E = E_new - E_curr
            if Delta_E < 0:
                E_curr = E_new
                p_curr = p_new.copy()
                if E_curr < E_best:
                    E_best = E_curr 
            else:
                Prob = math.exp(-Delta_E/T)
                r = random.uniform(0,1)
                if r < Prob:
                    E_curr = E_new
                    p_curr = p_new.copy()
            T = T * cooling_factor
            iter = iter + 1
            result.append(E_curr)
    print("Best solution :",E_best,"iter:",iter)
    #print("Minimum deger: ", min(result))
    s = list(range(0, iter))
    fig, ax = plt.subplots()
    ax.plot(s, result)
    ax.set(xlabel='Iteration Number', ylabel='Objective Value',title='Simulated Annealing for TSP')
    ax.grid()
    fig.savefig("test.png")
    plt.show()
            

x=[]
y=[]
p=[]
p,x,y=veri_oku(p,x,y)
xarray=np.asarray(x)
yarray=np.asarray(y)
xmutlak=abs(xarray)
ymutlak=abs(yarray)
print(xmutlak,ymutlak,p)
simulatedAnnealing(p,xmutlak,ymutlak,1000)