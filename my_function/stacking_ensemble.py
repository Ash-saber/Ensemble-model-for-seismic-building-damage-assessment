# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 12:09:34 2022

@author: Steve
"""

import copy
import numpy as np
from math import exp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
a = np.empty([0,2])
b = np.array([[1,1],[1,1]])
c = np.append(a,b,axis = 0)
y_metatrain = [np.empty([5,0]) for i in range (4)]

class stacking_ensemble():
    def __init__(
            self,
            model_list: list,
            multiclass=True):
        self.model_list = model_list
        self.multiclass = multiclass
    
    def _onehot_encode(self,y):
        y = y.reshape(-1)
        y = y.astype('int')
        y_max = np.max(y)
        y_encode = np.zeros((len(y),y_max+1)).astype('int')
        for i in range(len(y)):
            y_encode[i,y[i]] = 1
        return y_encode
    def train(self,x_smp,y_smp,one_hot:list):
        #x_smp, y_smp must be numpy array
            
#        if one_hot==True:
#            y_smp = self._onehot_encode(y_smp)
        self.one_hot = one_hot
        x_train, x_valid, y_train, y_valid = train_test_split(x_smp,y_smp,
                                                            test_size = 0.3,
                                                            random_state = 0)
        
        for i, model in enumerate(self.model_list):
            name = 'TabNetRegressor'
            name2 ='TabNetClassifier'

            if str(model.__str__)[29:44]==name or str(model.__str__)[29:45]==name2:
                x_t, x_v, y_t, y_v = train_test_split(x_train,y_train,
                                                                    test_size = 0.3,
                                                                    random_state = 0)
                model.fit(x_t,y_t,eval_set=[(x_v, y_v)])
            else:
                if one_hot[i]==False:
                    model.fit(x_train,y_train)
                else:
                    model.fit(x_train,self._onehot_encode(y_train))
            print(f'model_{i} has been trained')
        categ_num = len(np.unique(y_smp))
        model_num = len(self.model_list)
        self.categ_num = categ_num
        y_metatrain = [np.zeros([len(y_valid),model_num]) for i in range (categ_num)]
        for i, model in enumerate(self.model_list):
            if one_hot[i]==True:
                yp = model.predict(x_valid)
            else:
                yp = model.predict_proba(x_valid)
                
            for j in range(categ_num):
                y_metatrain[j][:,i] = yp[:,j]
        
        reg_list = [LinearRegression() for i in range(categ_num)]
        for i, x_metasmp in enumerate(y_metatrain):
            y_metasmp = self._onehot_encode(y_valid)[:,i].reshape(-1,1)
            reg_list[i].fit(x_metasmp,y_metasmp)
        self.reg_list = reg_list
    
    def predict(self,x_smp):
        model_num = len(self.model_list)
        y_pred = [np.zeros([len(x_smp),model_num]) for i in range (self.categ_num)]
        for i, model in enumerate(self.model_list):
            if self.one_hot[i]==False:
                yp = model.predict_proba(x_smp)
            else:
                yp = model.predict(x_smp)
            for j in range(self.categ_num):
                y_pred[j][:,i] = yp[:,j]
        y_metapred = np.zeros((len(x_smp),self.categ_num))  
        
        for i, reg_model in enumerate(self.reg_list):
            y_metapred[:,i] = reg_model.predict(y_pred[i]).reshape(-1)
        return y_metapred
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        