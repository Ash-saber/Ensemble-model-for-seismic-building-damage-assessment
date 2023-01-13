# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 14:11:44 2023

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)

from my_function.my_lgb_and_xgb import xgb_ensemble,lgb_ensemble
import numpy as np
from math import exp


from my_function.stacking_ensemble import stacking_ensemble
from my_function.my_saint import *
from my_function.my_tabtransformer import *
from sklearn.ensemble import RandomForestClassifier as rfc
from pytorch_tabnet.tab_model import TabNetClassifier as tabc
def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc

data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

#data_x = np.delete(data_x,[5,14,9,10],axis=1)

extreme_index = np.where(data_x[:,2]==999)[0]
data_x[extreme_index,2]=200

def percent_cal(x_smp):
    a = np.unique(x_smp)
    p = [np.sum(x_smp==a[i])/len(x_smp) for i in range(len(a))]
    return p

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data_x, data_y,
                                                    test_size = 0.5,
                                                    random_state = 1)   
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train,
                                                    test_size = 0.3,
                                                    random_state = 0) 

import pickle

    
with open ('model_save/stack_model_c.pkl','rb') as f:
    stack_model_6 = pickle.load(f)


with open ('model_save/stack_model_c_3.pkl','rb') as f:
    stack_model_3 = pickle.load(f)


coef_3 = np.array([stack_model_3.reg_list[i].coef_.reshape(-1) for i in range(5)])
coef_6 = np.array([stack_model_6.reg_list[i].coef_.reshape(-1) for i in range(5)])



acc_s_t3 = acc_cal(stack_model_3.predict(x_t).argmax(1),y_t)
acc_s_v3 = acc_cal(stack_model_3.predict(x_v).argmax(1),y_v)

import matplotlib.pyplot as plt 


plt.rc('font',family='Times New Roman')
figure = plt.figure(dpi= 500) 
ax = figure.add_subplot(111) 
ax.matshow(coef_6, interpolation ='nearest',cmap='RdBu') 
for i in range(coef_6.shape[0]):
    for j in range(coef_6.shape[1]):
        ax.annotate("%.2f" % coef_6[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center',
                     color = 'black' if coef_6[i, j]>0 and coef_6[i, j]<0.5 else 'white')
y_text = [f'DG{i+1}' for i in range(5)]
ax.set_yticklabels(['']+y_text) 
x_text = ['TabNet','XGBoost','SAINT','LGBM','Tab-trans','RF']
ax.set_xticklabels(['']+x_text) 
ax.set_title('Accuracy: 70.8% (training) 50.7% (validation)',y=-0.1)
plt.savefig('img/initial_stacking.svg',bbox_inches='tight',dpi=600)

figure = plt.figure(dpi= 500) 
ax2 = figure.add_subplot(111) 
ax2.matshow(coef_3, interpolation ='nearest',cmap='Blues') 
for i in range(coef_3.shape[0]):
    for j in range(coef_3.shape[1]):
        ax2.annotate("%.2f" % coef_3[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center',
                     color = 'black' if coef_3[i, j]>0 and coef_3[i, j]<0.4 else 'white')
y_text = [f'DG{i+1}' for i in range(5)]
ax2.set_yticklabels(['']+y_text) 
x_text = ['XGBoost','SAINT','RF']
ax2.set_xticklabels(['']+x_text) 
ax2.set_title('Accuracy: 75.3% (training) 51.4% (validation)',y=-0.1)
plt.savefig('img/optimized_stacking.svg',bbox_inches='tight',dpi=600)



