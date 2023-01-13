# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:14:43 2023

@author: Steve
"""
import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from my_function.fuzzy_predict import *
import shap
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

#data_x = np.delete(data_x,[5,14,9,10],axis=1)

extreme_index = np.where(data_x[:,2]==999)[0]
data_x[extreme_index,2]=200

x_train, x_valid, y_train, y_valid = train_test_split(data_x, data_y,
                                                    test_size = 0.5,
                                                    random_state = 1)   


def acc_cal(yp,yt):
    acc = np.sum(yp==yt)/len(yt)
    return acc
def percent_cal(x_smp):
    a = np.unique(x_smp)
    p = [np.sum(x_smp==a[i])/len(x_smp) for i in range(len(a))]
    return p
def precision_cal(c_mat):
    p_sum = np.sum(c_mat,axis=0)
    p_vector = [c_mat[i,i]/p_sum[i] for i in range(len(c_mat))]
    return np.array(p_vector)

def recall_cal(c_mat):
    t_sum = np.sum(c_mat,axis=1)
    r_vector = [c_mat[i,i]/t_sum[i] for i in range(len(c_mat))]
    return np.array(r_vector)

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
with open ('model_save/boost_model_3.pkl','rb') as f:
    boost_model = pickle.load(f)   
with open('feature_names.txt', "r", encoding='utf-8') as f: 
    feature_names = f.read()   
feature_names =feature_names.split()
class_names = [f'Damage Grade {i+1}' for i in range(5)]

color_list = lambda i: ['#845ec2','#4b4453','#b0a8b9','#c34a36','#ff8066'][i]
color_list = lambda i: ['#0339A6','#04ADBF','#F2B705','#F28705','#F2380F'][i]

f = lambda x: softmax(boost_model.predict(x))
med = np.median(x_valid,axis=0).reshape(1,-1)
explainer = shap.KernelExplainer(f, med)
shap_values = explainer.shap_values(x_valid[0:1000])

feature_order = np.argsort(-np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
new_feature = np.arange(21)
for i in range(21):
    new_feature[feature_order[i]]=i+1
new_feature = new_feature.astype('str')

fig = plt.figure(dpi=600)
plt.rc('font',family='Times New Roman')
shap.summary_plot(shap_values=shap_values,
                  features = x_valid[0:1000], 
                  feature_names=new_feature,
                  class_names = class_names,
                  color=color_list,
                  plot_type = 'bar',
                 max_display=10,
                 show=False)
#plt.xticks(fontsize=14)
plt.yticks(weight='bold')
plt.xlabel('Average impact on damage grade prediction')

for i in range(10):
    plt.text(0.05,7.5-i*0.6,f'{i+1}. {feature_names[feature_order[i]]}',fontsize=14)

plt.savefig('img/shap_importance.svg',bbox_inches='tight',dpi=600)


fig = plt.figure(dpi=600)
plt.axis('off') 
plt.rc('font',family='Times New Roman')
for i in range(10):
    plt.text(0.2,0.95-0.1*i,f'{i+1}. {feature_names[feature_order[i]]}',fontsize=14,weight='bold')
plt.text(0.2,0.95-0.1*10,f'{11}. {feature_names[feature_order[11]]}',fontsize=14,weight='bold')
plt.savefig('img/shap_summary_legend.svg',bbox_inches='tight',dpi=600)    


f = lambda x: softmax(boost_model.predict(x))[:,4]
med = np.median(x_valid,axis=0).reshape(1,-1)
explainer = shap.KernelExplainer(f, med)
shap_values = explainer.shap_values(x_valid[0:1000])

shape_value_list =[]
for i in range(5):
    f = lambda x: softmax(boost_model.predict(x))[:,i]
    med = np.median(x_valid,axis=0).reshape(1,-1)
    explainer = shap.KernelExplainer(f, med)
    shape_value_list.append(explainer.shap_values(x_valid[0:1000]))
    

plt.rc('font',family='Times New Roman')
plt.rcParams['font.size']=17


i=4
shap.summary_plot(shap_values=shape_value_list[i],
                  features = x_valid[0:1000], 
                  feature_names=new_feature,
                  class_names = class_names,
                  plot_type = 'dot',
                 max_display=10,
                 show=False)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17,weight='bold')
plt.xlabel('SHAP value',fontsize=17)


plt.savefig(f'img/SHAP_summary_plots_DG{i+1}.svg',bbox_inches='tight',dpi=600)    
    
    
    
    
    
    
    
    
    