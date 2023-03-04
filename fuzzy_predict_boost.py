# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:47:11 2023

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from my_function.fuzzy_predict import *
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
    
with open ('model_save/boost_model_3.pkl','rb') as f:
    boost_model = pickle.load(f)   
yp_boost = boost_model.predict(x_valid).argmax(1)
yp_boost_p = softmax(boost_model.predict(x_valid))
inacc = Interval_acc(yp_boost_p,y_valid)
interval,fuzzy_yp = inacc.interval_cal()
acc = acc_cal(y_valid,yp_boost)

acc_f = inacc.acc()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

i1=3114
i2=3134
i1=21100
i2=21130
smp = interval[i1:i2,:]
smp_t = y_valid[i1:i2]+1
up_limit = np.array([np.max(smp[i,0:2]) for i in range(len(smp))]).astype('int')
low_limit = np.array([np.min(smp[i,0:2]) for i in range(len(smp))]).astype('int')
sort = np.argsort(up_limit)
up_limit = up_limit[sort]
low_limit = low_limit[sort]
smp = smp[sort]
smp_t = smp_t[sort]

distinct_smp=smp[:,0]+1


color_list=['#845ec2','#008c82']
fig = plt.figure(dpi=600,figsize=(12,5))
ax = fig.add_subplot(122) 
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size']=12
ax.plot(up_limit+1,lw=1.5,marker='o',label='upper damage grade',markersize=4,color=color_list[0])
ax.plot(low_limit+1,lw=1.5,marker='o',label='lower damage grade',markersize=4,color=color_list[1])
ax.plot(smp_t,lw=2,label='true damage grade',color = 'black',linestyle='--',alpha=0.6,marker='o',markersize=4)
for i in range(len(smp)):
    start = smp[i,2] if smp[i,0]<smp[i,1] else smp[i,3]
    left, bottom, width, height = (i-0.2, low_limit[i]+1, 0.4, start)
    rect=mpatches.Rectangle((left,bottom),width,height, 
                           linewidth=2,                        
                           alpha=0.3,
                           facecolor=color_list[1])
    plt.gca().add_patch(rect)
    end = smp[i,2] if smp[i,0]>smp[i,1] else smp[i,3]
    left, bottom, width, height = (i-0.2, low_limit[i]+1+start, 0.4, end)
    rect=mpatches.Rectangle((left,bottom),width,height, 
                           linewidth=2,                        
                           alpha=0.3,
                           facecolor=color_list[0])
    plt.gca().add_patch(rect)
ax.legend(['predicted upper', 'predicted lower','true damage grade','lower probability',
           'upper probability'],
          loc='lower right', frameon=False)
ax.set_yticks(np.arange(5)+1)
ax.set_xlabel('Sample number')
ax.set_ylabel('Damage grade')
ax.set_title('(b) fuzzy prediction',y=-0.2)


ax2 = fig.add_subplot(121) 
ax2.plot(distinct_smp,lw=2,label='predicted damage grade',color = '#c34a36',linestyle='-',marker='o',markersize=4)
ax2.plot(smp_t,lw=2,label='true damage grade',color = 'black',linestyle='--',alpha=0.6,marker='o',markersize=4)
ax2.legend(loc='lower right', frameon=False)
ax2.set_yticks(np.arange(5)+1)
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Damage grade')
ax2.set_title('(a) distinct prediction',y=-0.2)
plt.savefig('img/fuzzy_prediction.svg',bbox_inches='tight',dpi=600)






