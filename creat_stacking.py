# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:40:30 2022

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'# working path
os.chdir(path)
import pickle
from my_function.my_lgb_and_xgb import xgb_ensemble,lgb_ensemble
import numpy as np
from math import exp
from xgboost import XGBClassifier as xgbc
from lightgbm import LGBMClassifier as lgbc

from my_function.stacking_ensemble import stacking_ensemble
from my_function.my_saint import *
from my_function.my_tabtransformer import *
from sklearn.ensemble import RandomForestClassifier as rfc
from pytorch_tabnet.tab_model import TabNetClassifier as tabc
def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc
def percent_cal(x_smp):
    a = np.unique(x_smp)
    p = [np.sum(x_smp==a[i])/len(x_smp) for i in range(len(a))]
    return p
data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

#data_x = np.delete(data_x,[5,14,9,10],axis=1) 

extreme_index = np.where(data_x[:,2]==999)[0] # modify extreme value
data_x[extreme_index,2]=200







from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data_x, data_y,
                                                    test_size = 0.5,
                                                    random_state = 1)    
#from imblearn.under_sampling import RandomUnderSampler
#rus=RandomUnderSampler(random_state=0)
#x_res,y_res=rus.fit_sample(x_train,y_train)

# creat model list for stacking model
model_list = [tabc(),xgb_ensemble(),saint_model(),lgb_ensemble(),tab_transformer(),rfc()]
one_hot_list = [0,1,0,1,0,0]
#model_list=[tab_transformer(),xgbc()]
stack_model_6 = stacking_ensemble(model_list)
stack_model_6.train(x_train,y_train,one_hot=one_hot_list)

coef = np.array([stack_model_6.reg_list[i].coef_.reshape(-1) for i in range(5)])# check the coefficient

yp = stack_model_6.predict(x_train).argmax(1)
acc1 = acc_cal(yp,y_train)

yp = stack_model_6.predict(x_valid).argmax(1)
acc2 = acc_cal(yp,y_valid)

yp = stack_model_6.model_list[1].predict(x_valid).argmax(1)
acc4 = acc_cal(yp,y_valid)

with open ('model_save/stack_model_6.pkl','wb') as f:
    pickle.dump(stack_model_6,f)


model_list = [xgbc(),saint_model(epochs=10),rfc()]# reselect sub-model
stack_model_c_3 = stacking_ensemble(model_list)
stack_model_c_3.train(x_train,y_train,one_hot=[0,0,0])

coef = np.array([stack_model_c_3.reg_list[i].coef_.reshape(-1) for i in range(5)])

yp = stack_model_c_3.predict(x_train).argmax(1)
acc1 = acc_cal(yp,y_train)

yp = stack_model_c_3.predict(x_valid).argmax(1)
acc2 = acc_cal(yp,y_valid)

with open ('model_save/stack_model_c_3.pkl','wb') as f:
    pickle.dump(stack_model_c_3,f)















