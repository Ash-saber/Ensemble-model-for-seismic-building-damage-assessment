# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:43:30 2022

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)

from xgboost import XGBRegressor as xgbr
from lightgbm import LGBMRegressor as lgbr
import numpy as np
from math import exp
import pickle
from my_function.my_lgb_and_xgb import xgb_ensemble,lgb_ensemble
from my_function.boosting_ensemble import boosting_ensemble
from my_function.my_saint import *
from my_function.my_tabtransformer import *
from sklearn.ensemble import RandomForestRegressor as rf
from pytorch_tabnet.tab_model import TabNetRegressor as tabr
def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc
def onehot_encode(y):
    y = y.reshape(-1)
    y = y.astype('int')
    y_max = np.max(y)
    y_encode = np.zeros((len(y),y_max+1)).astype('int')
    for i in range(len(y)):
        y_encode[i,y[i]] = 1
    return y_encode
#data_y = onehot_encode(data_y)
data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1
extreme_index = np.where(data_x[:,2]==999)[0] # modify extreme value
data_x[extreme_index,2]=200



from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data_x, onehot_encode(data_y),
                                                    test_size = 0.5,
                                                    random_state = 1)    

model_list = [xgb_ensemble(),saint_model(multiclass=False),lgb_ensemble(),tabr(),tab_transformer(multiclass=False),rf()]
#model_list = [xgb_ensemble(),lgb_ensemble()]
boost_model = boosting_ensemble(model_list,learning_rate=[1,0.5,0.5,0.5,0.5,0.5])
boost_model.train(x_train,y_train)
a = boost_model.boosting_acc
yp = boost_model.predict(x_valid)
acc = acc_cal(yp.reshape(-1),y_valid.argmax(1))
with open ('model_save/boost_model_6.pkl','wb') as f:
    pickle.dump(boost_model,f)



model_list2 = [xgb_ensemble(),saint_model(multiclass=False,epochs=12),rf()]
#model_list = [xgb_ensemble(),lgb_ensemble()]
boost_model2 = boosting_ensemble(model_list2,learning_rate=[1,0.5,0.2])
boost_model2.train(x_train,y_train)
a2 = boost_model2.boosting_acc

yp2 = boost_model2.predict(x_valid)

acc2 = acc_cal(yp2.reshape(-1),y_valid.argmax(1))

with open ('model_save/boost_model_3.pkl','wb') as f:
    pickle.dump(boost_model2,f)


yp = boost_model.model_list[0].predict(x_valid).argmax(1)
acc3 = acc_cal(yp.reshape(-1,1),y_valid)

yp = boost_model.model_list[0].predict(x_train).argmax(1)
acc4 = acc_cal(yp.reshape(-1,1),y_train)

a = boost_model.boosting_acc

with open ('model_save/boost_model_6model.pkl','wb') as f:
    pickle.dump(boost_model,f)









