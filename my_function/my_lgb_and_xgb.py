# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:41:19 2022

@author: Steve
"""
from xgboost import XGBRegressor as xgbr
from lightgbm import LGBMRegressor as lgbr
import numpy as np
from math import exp

class xgb_ensemble():

    def fit(self,x_smp,y_smp):
        self.y = [y_smp[:,i].reshape(-1) for i in range(y_smp.shape[1])]
        self.x = x_smp
        self.module_list = [xgbr() for i in range(y_smp.shape[1])]
        self.module_len = len(self.module_list)
        for i in range(self.module_len):
            self.module_list[i].fit(self.x,self.y[i])
            
    def predict(self,new_x):
        y_predict_list  = [self.module_list[i].predict(new_x) for i in range(self.module_len)]
        tmp = y_predict_list[0].reshape(-1,1)
        for i in range(1,self.module_len):
            tmp = np.concatenate((tmp,y_predict_list[i].reshape(-1,1)),1)
        y_predict = tmp
        return y_predict
    
class lgb_ensemble():

    def fit(self,x_smp,y_smp):
        self.y = [y_smp[:,i].reshape(-1) for i in range(y_smp.shape[1])]
        self.x = x_smp
        self.module_list = [lgbr() for i in range(y_smp.shape[1])]
        self.module_len = len(self.module_list)
        for i in range(self.module_len):
            self.module_list[i].fit(self.x,self.y[i])
            
    def predict(self,new_x):
        y_predict_list  = [self.module_list[i].predict(new_x) for i in range(self.module_len)]
        tmp = y_predict_list[0].reshape(-1,1)
        for i in range(1,self.module_len):
            tmp = np.concatenate((tmp,y_predict_list[i].reshape(-1,1)),1)
        y_predict = tmp
        return y_predict


        