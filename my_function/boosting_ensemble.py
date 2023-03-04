# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:35:33 2022

@author: Steve
"""

import copy
import numpy as np
from math import exp
from sklearn.model_selection import train_test_split

def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc

class boosting_ensemble():
    def __init__(
            self,
            model_list: list,
            learning_rate: list = None):
        self.model_list = model_list
        if learning_rate == None:
            learning_rate = [0.2 for i in range(len(model_list))]
        self.learning_rate = learning_rate
        self.train_accomplish = False

        
    def _softmax(self,y):
        y = y.reshape(-1)
        temp0 = np.array([exp(y[i]) for i in range(len(y))])
        y_sum = np.sum(temp0)
        y_new = temp0/y_sum
        y_new = y_new.reshape(1,-1)
        return y_new
    
    def _residual_cal(self,yp,yt):
        residual = np.zeros(yt.shape)
        for i in range(len(yt)):
            residual[i,:] = yt[i]-self._softmax(yp[i])
        return residual
    
    def _onehot_encode(self,y):
        y = y.reshape(-1)
        y = y.astype('int')
        y_max = np.max(y)
        y_encode = np.zeros((len(y),y_max+1)).astype('int')
        for i in range(len(y)):
            y_encode[i,y[i]] = 1
        return y_encode
    
    def train(self,x_smp,y_smp,one_hot = False):
        #x_smp, y_smp must be numpy array
            
        if one_hot==True:
            y_smp = self._onehot_encode(y_smp)
            
        x_train, x_valid, y_train, y_valid = train_test_split(x_smp, y_smp,
                                                            test_size = 0.3,
                                                            random_state = 1) 
        boosting_acc = []
        for i, model in enumerate(self.model_list):
            learning_rate = self.learning_rate[i]
                
            if i ==0:
                model.fit(x_train,y_train)
                print(f'model_{i} has been trained')
                yp_train = model.predict(x_train).argmax(1)
                acc_train = acc_cal(yp_train.reshape(-1),y_train.argmax(1))
                yp_valid = model.predict(x_valid).argmax(1)
                acc_valid = acc_cal(yp_valid.reshape(-1),y_valid.argmax(1))
                print('[**Boosting step %d**]: [Train]: %.3f]   [Valid]: %.3f' % (i,acc_train,acc_valid))
                boosting_acc.append([acc_train,acc_valid])
            else:
                for j in range(i):
                    if j == 0:
                        last_y_prob = self.model_list[j].predict(x_train)
                    else:
                        last_y_prob = self.model_list[j].predict(x_train)+last_y_prob
                residual = self._residual_cal(last_y_prob, y_train)
                name1 = 'TabNetRegressor'
                name2 ='TabNetClassifier'
                if str(model.__str__)[29:44]==name1 or str(model.__str__)[29:45]==name2:
                    x_t, x_v, y_t, y_v = train_test_split(x_train,learning_rate*residual,
                                                                        test_size = 0.3,
                                                                        random_state = 0)
                    model.fit(x_t,y_t,eval_set=[(x_v, y_v)])
                else:
                    model.fit(x_train,learning_rate*residual)
                
                
                for j in range(i+1):
                    if j == 0:
                        yp_train = self.model_list[j].predict(x_train)
                        yp_valid = self.model_list[j].predict(x_valid)
                    else:
                        yp_train = self.model_list[j].predict(x_train)+yp_train
                        yp_valid = self.model_list[j].predict(x_valid)+yp_valid
                print(f'model_{i} has been trained')
                yp_train = yp_train.argmax(1)
                yp_valid = yp_valid.argmax(1)
                acc_train = acc_cal(yp_train.reshape(-1),y_train.argmax(1))
                acc_valid = acc_cal(yp_valid.reshape(-1),y_valid.argmax(1))
                print('[**Boosting step %d**: [Train]: %.3f]   [Valid]: %.3f' % (i,acc_train,acc_valid))
                boosting_acc.append([acc_train,acc_valid])
        self.train_accomplish = True
        self.boosting_acc=np.array(boosting_acc)
    
    def predict(self,x_smp):
        if self.train_accomplish == True:
            for i, model in enumerate(self.model_list):
                if i == 0:
                    yp = model.predict(x_smp)
                else:
                    yp = model.predict(x_smp)+yp
            yp = yp
            return yp
        else:
            print('Training process has not been done!')
            
        
            
            
            
            
            
            
            
            
            
            
            