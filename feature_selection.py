# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:26:16 2022

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
from pytorch_tabnet.tab_model import TabNetClassifier as tabc
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import pandas as pd

data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

scalar_x = MinMaxScaler()
scalar_x=scalar_x.fit(data_x)
x_smp = scalar_x.transform(data_x)

y_smp = data_y

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_smp, y_smp,
                                                    test_size = 0.5,
                                                    random_state = 1)    

def miv_cal(model,x_smp):
    smp_len = x_smp.shape[0]
    fea_len = x_smp.shape[1]
    
    model = my_tabc
    y_old = model.predict(x_smp)
    y_old_prob = model.predict_proba(x_smp)
    y_old_oneprob = [y_old_prob[i,y_old[i]] for i in range(smp_len)]
    y_old_oneprob = np.array(y_old_oneprob)

    x_unique = []
    for i in range(x_smp.shape[1]):
        xi_unique = np.unique(x_smp[:,i])
        x_unique.append(xi_unique)

    x_new_set = [] #generate new x_smp
    miv = []
    for i in range(fea_len):
        unique = x_unique[i]
        xi_new = copy.deepcopy(x_smp)
        xi_new[:,i] = np.random.choice(unique,smp_len)
        yi_new_prob = model.predict_proba(xi_new)
        yi_new_oneprob = np.array([yi_new_prob[i,y_old[i]] for i in range(smp_len)])
        miv_xi = np.mean(np.abs(yi_new_oneprob-y_old_oneprob))
        miv.append(miv_xi)
    miv = np.array(miv)
    return miv

#miv
my_tabc = tabc()
my_tabc.fit(x_train,y_train,eval_set=[(x_valid, y_valid)])
miv = miv_cal(my_tabc,x_train)   
    

#GRA
from my_function.GRA import GRA_ONE

temp = pd.DataFrame(np.concatenate((x_train,y_train.reshape(-1,1).astype('float')),axis=1))
grg = GRA_ONE(temp,m=21)

    
#vim
rnd_clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1, random_state= 1)
rnd_clf.fit(x_train,y_train)

vim = rnd_clf.feature_importances_    
    
var_list = [f'A{i+1}' for i in range(10) ]+[f'B{i+1}' for i in range(11)]

#pearson
m_pearson = temp.corr(method = 'pearson')
pearson = abs(np.array(m_pearson.iloc[21,:21]))

#normalize and rank
def norm(series):
    s_sum = np.sum(series)
    norm_series = series/s_sum
    return norm_series
grg = norm(grg)
vim = norm(vim)
miv = norm(miv)
pearson = norm(pearson)
hybrid = (vim+miv+pearson)/3

thres = np.percentile(hybrid, 10)
np.sum(hybrid>thres)

hy_sort = np.argsort(-hybrid)

cm_hy = np.cumsum(norm(hybrid)[hy_sort])

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



fig = plt.figure(dpi=600,figsize=(8.5,5))
gs = plt.GridSpec(7,11)   # 
plt.rc('font',family='Times New Roman')
plt.rcParams['font.size']=12
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
ax1 = fig.add_subplot(gs[4:7,0:11])  
ax2 = fig.add_subplot(gs[0:3,0:3])
ax3 = fig.add_subplot(gs[0:3,4:7])
ax4 = fig.add_subplot(gs[0:3,8:11])


color_list = ['#d62728']*17+['#1f77b4']*4
#ax1.bar(np.array(var_list)[hy_sort],hybrid[hy_sort],width=0.55
       #,alpha=0.8,color = color_list)
ax1.bar(np.array(var_list)[hy_sort[0:17]],hybrid[hy_sort[0:17]],width=0.55
       ,alpha=0.8,color = color_list[0:17])
ax1.bar(np.array(var_list)[hy_sort[17:]],hybrid[hy_sort[17:]],width=0.55
       ,alpha=0.8,color = color_list[17:])
ax1.legend(['Selected','Eliminated'],loc = 'upper right',frameon=False)
ax1.set_xlabel('Ranked feature')
ax1.set_ylabel('Hybrid coefficient')
ax1.set_title('(d)',y=-0.4)
ax1.vlines(16.5, 0, 0.14, linestyles='dashed', colors='darkred')
ax1.text(12,0.1,'95% accumulation',Fontsize=12)
#ax12 = ax1.twinx()
#ax12.plot(cm_hy)


ax2.bar(var_list,pearson,width=0.55
       ,alpha=0.8)
ax2.set_xticks(np.arange(21,step=5))
ax2.set_xlabel('Feature')
ax2.set_ylabel('PCC')
ax2.set_title('(a)',y=-0.4)

ax3.bar(var_list,vim,width=0.55
       ,alpha=0.8)
ax3.set_xticks(np.arange(21,step=5))
ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax3.set_xlabel('Feature')
ax3.set_ylabel('VIM')
ax3.set_title('(b)',y=-0.4)

ax4.bar(var_list,miv,width=0.55
       ,alpha=0.8)
ax4.set_xticks(np.arange(21,step=5))
ax4.set_xlabel('Feature')
ax4.set_ylabel('MIV')
ax4.set_title('(c)',y=-0.4)
plt.savefig('img/feature_selsection.svg',bbox_inches='tight',dpi=600)

