# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:10:10 2023

@author: Steve
"""



from math import exp
import numpy as np
def softmax(y):
    y_softmax = []
    for i in range(len(y)):
        y_temp = y[i,:]
        temp0 = np.array([exp(y_temp[j]) for j in range(len(y_temp))])
        y_sum = np.sum(temp0)
        y_new = temp0/y_sum
        y_new = y_new
        y_softmax.append(y_new)
    return np.array(y_softmax)

class Interval_acc():
    def __init__(self,output,y_true):
        self.output = output
        self.x = output.argmax(1)
        self.y_true = y_true
    def interval_cal(self):
        output = self.output 
        x = self.x
        interval = np.ones((output.shape[0],4))
        fuzzy_prediction = []
        for i in range(len(output)):
            prob_xi = output[i,x[i]]
            prob_xil = output[i,x[i]-1]
            if x[i]>0 and x[i]<4:
                prob_xir = output[i,x[i]+1]
                
        
                if prob_xil>prob_xir:
                    interval[i,0] = x[i]
                    interval[i,1] = x[i]-1
                    prob1 = prob_xi/(prob_xil+prob_xi)
                    prob2 = prob_xil/(prob_xil+prob_xi)
                    interval[i,2] = prob1
                    interval[i,3] = prob2
                else:
                    interval[i,0] = x[i]
                    interval[i,1] = x[i]+1
                    prob1 = prob_xi/(prob_xir+prob_xi)
                    prob2 = prob_xir/(prob_xir+prob_xi)
                    interval[i,2] = prob1
                    interval[i,3] = prob2
            elif x[i] ==0:
                prob_xi = output[i,x[i]]
                prob_xir = output[i,x[i]+1]
                interval[i,0] = x[i]
                interval[i,1] = x[i]+1
                prob1 = prob_xi/(prob_xir+prob_xi)
                prob2 = prob_xir/(prob_xir+prob_xi)
                interval[i,2] = prob1
                interval[i,3] = prob2
            elif x[i] ==4:
                prob_xi = output[i,x[i]]
                prob_xil = output[i,x[i]-1]
                interval[i,0] = x[i]
                interval[i,1] = x[i]-1
                prob1 = prob_xi/(prob_xil+prob_xi)
                prob2 = prob_xil/(prob_xil+prob_xi)
                interval[i,2] = prob1
                interval[i,3] = prob2
            fuzzy_yp = np.random.choice(interval[i,0:2],1,p=interval[i,2:4])
            fuzzy_prediction.append(fuzzy_yp)
        self.interval = interval
        self.fuzzy_prediction = np.array(fuzzy_prediction).astype('int')
        return self.interval,self.fuzzy_prediction
    
    def acc(self):
        interval = self.interval
        y_true = self.y_true
        y_true = y_true.reshape(-1,1)
        y = np.repeat(y_true,4,axis=1)
        self.acc = (interval == y).sum()/len(y_true)    
        return self.acc
    
if __name__ == '__main__':
    a = np.ones(10)*2
    b = np.random.randn(10,5)
    b = softmax(b)
    c = b.argmax(1)
    inacc = Interval_acc(b,a)
    k,fuzzy_yp = inacc.interval_cal()
    k1 = inacc.acc()
    print(k1)
    print(k)
    print(c)
    print(b)