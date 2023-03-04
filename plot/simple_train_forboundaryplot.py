# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:39:00 2023

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
import numpy as np
from my_function.stacking_ensemble import stacking_ensemble
from my_function.my_lgb_and_xgb import xgb_ensemble,lgb_ensemble
from my_function.boosting_ensemble import boosting_ensemble
from my_function.my_saint import *
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import RandomForestClassifier as rfc

from xgboost import XGBClassifier as xgbc
import pickle
data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

#data_x = np.delete(data_x,[5,14,9,10],axis=1)

extreme_index = np.where(data_x[:,2]==999)[0]
data_x[extreme_index,2]=200

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
x_res,y_res=rus.fit_sample(data_x,data_y)
x_res = x_res[:,2:4]


def onehot_encode(y):
    y = y.reshape(-1)
    y = y.astype('int')
    y_max = np.max(y)
    y_encode = np.zeros((len(y),y_max+1)).astype('int')
    for i in range(len(y)):
        y_encode[i,y[i]] = 1
    return y_encode

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_res, y_res,
                                                    test_size = 0.5,
                                                    random_state = 1)    


model_list = [xgb_ensemble(),saint_model(multiclass=False,epochs=12),rf()]
#model_list = [xgb_ensemble(),lgb_ensemble()]
boost_model_simple = boosting_ensemble(model_list,learning_rate=[1,0.5,0.2])
boost_model_simple.train(x_train,onehot_encode(y_train))
with open ('model_save/boost_model_simple.pkl','wb') as f:
    pickle.dump(boost_model_simple,f)


model_list2 = [xgbc(),saint_model(epochs=10),rfc()]
stack_model_simple = stacking_ensemble(model_list2)
stack_model_simple.train(x_train,y_train,one_hot=[0,0,0])

with open ('model_save/stakc_model_simple.pkl','wb') as f:
    pickle.dump(stack_model_simple,f)


