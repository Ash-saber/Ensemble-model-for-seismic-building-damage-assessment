# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:29:24 2022

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

from xgboost import XGBRegressor as xgbr
from lightgbm import LGBMRegressor as lgbr
import numpy as np
from math import exp

from my_function.my_lgb_and_xgb import xgb_ensemble,lgb_ensemble
from my_function.stacking_ensemble import stacking_ensemble
from my_function.my_saint import *
from my_function.my_tabtransformer import *
from sklearn.ensemble import RandomForestRegressor as rf
from pytorch_tabnet.tab_model import TabNetRegressor as tabr

def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc

data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1
#data_x = np.delete(data_x,[5,14,9,10],axis=1)

def onehot_encode(y):
    y = y.reshape(-1)
    y = y.astype('int')
    y_max = np.max(y)
    y_encode = np.zeros((len(y),y_max+1)).astype('int')
    for i in range(len(y)):
        y_encode[i,y[i]] = 1
    return y_encode
#data_y = onehot_encode(data_y)

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data_x, data_y,
                                                    test_size = 0.5,
                                                    random_state = 1)    


with open ('model_save/stack_model.pkl','rb') as f:
    stack_model = pickle.load(f)

yp = stack_model.predict(x_valid).argmax(1)
acc2 = acc_cal(yp,y_valid)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
y_true = y_valid
c = confusion_matrix(y_true,yp)
norm_c = c/np.sum(c,axis=0) 



fig = plt.figure(dpi= 500)#
ax = fig.add_subplot(111)#
im = ax.matshow(norm_c, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
for i in range(len(norm_c)):
    for j in range(len(norm_c)):
        plt.annotate("%.2f" % norm_c[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center',
                     color = 'white'if i==j else 'black')
ax.text(2,-1,'Predicted DG',ha='center')
ax.set_ylabel('True DG')



















