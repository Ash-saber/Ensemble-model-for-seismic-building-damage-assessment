# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:48:07 2022

@author: Steve
"""

import numpy as np
import matplotlib.pyplot as plt

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
import pickle
with open ('model_save/stakc_model_simple.pkl','rb') as f:
    stack_model_simple = pickle.load(f)
with open ('model_save/boost_model_simple.pkl','rb') as f:
    boost_model_simple = pickle.load(f)

data_x = np.load('x_allsmp.npy')
data_y = np.load('y_allsmp.npy')-1

x1 = np.unique(data_x[:,2])
x2 = np.unique(data_x[:,3])
x5 = np.unique(data_x[:,4])
x7 = np.unique(data_x[:,6])

x = np.arange(0,201,1)

y = np.arange(70,5001,50)


#x = np.unique(data_x[:,0])
#y = np.array([np.unique(data_x[:,3])[i] for i in range(0,2129,20)]).astype('int')
xx,yy = np.meshgrid(x,y)
xy=np.c_[xx.ravel(), yy.ravel()]

z_simp_stack = stack_model_simple.predict(xy).argmax(1)

z_simp_boost = boost_model_simple.predict(xy).argmax(1)

z_rf = stack_model_simple.model_list[2].predict(xy)

z_saint = stack_model_simple.model_list[1].predict_proba(xy).argmax(1)

z_xgb = stack_model_simple.model_list[0].predict(xy)

color_list = ['#2E639B','#7EABCC','#F8F8F8','#D58979','#9D263B']

    
from matplotlib.colors import LinearSegmentedColormap

my_cmap = LinearSegmentedColormap.from_list('mycmap', color_list)


plt.contourf(xx, yy, z_simp_stack.reshape(xx.shape)+1,alpha=0.9,cmap=my_cmap)
plt.colorbar()
plt.show()

plt.contourf(xx, yy, z_simp_boost.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
plt.show()

plt.contourf(xx, yy, z_saint.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
plt.show()
plt.contourf(xx, yy, z_rf.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
plt.show()
plt.contourf(xx, yy, z_xgb.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
plt.show()

from matplotlib import mathtext

mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'mathtext.default': 'default',
                     'mathtext.fontset': 'stix',
                     'font.family': 'Times New Roman'})

plt.rc('font',family='Times New Roman')



fig = plt.figure(dpi=600,figsize=(7.5,5))
gs = plt.GridSpec(27,72)   
ax1 = fig.add_subplot(gs[0:11,0:20])  
ax2 = fig.add_subplot(gs[0:11,26:46])
ax3 = fig.add_subplot(gs[0:11,52:72])
ax4 = fig.add_subplot(gs[16:27,13:33])
ax5 = fig.add_subplot(gs[16:27,39:59])

plt.rcParams['font.size']=10
ax1.contourf(xx, yy/1000, z_xgb.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
ax1.set_xlabel('Age of building (year)')
ax1.set_ylabel('Plinth area (×$10^{3}$ $ft^{2}$)')
ax1.set_title('(a) XGBoost',y=-0.45,fontsize=10)

ax2.contourf(xx, yy/1000, z_saint.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
ax2.set_xlabel('Age of building (year)')
ax2.set_ylabel('Plinth area (×$10^{3}$ $ft^{2}$)')
ax2.set_title('(b) SAINT',y=-0.45,fontsize=10)

ax3.contourf(xx, yy/1000, z_rf.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
ax3.set_xlabel('Age of building (year)')
ax3.set_ylabel('Plinth area (×$10^{3}$ $ft^{2}$)')
ax3.set_title('(c) Random Forest',y=-0.45,fontsize=10)

ax4.contourf(xx, yy/1000, z_simp_stack.reshape(xx.shape),alpha=0.9,cmap=my_cmap)
ax4.set_xlabel('Age of building (year)')
ax4.set_ylabel('Plinth area (×$10^{3}$ $ft^{2}$)')
ax4.set_title('(d) Stacking',y=-0.45,fontsize=10)

im=ax5.contourf(xx, yy/1000, z_simp_boost.reshape(xx.shape)+1,alpha=0.9,cmap=my_cmap)
ax5.set_xlabel('Age of building (year)')
ax5.set_ylabel('Plinth area (×$10^{3}$ $ft^{2}$)')
ax5.set_title('(e) Boosting',y=-0.45,fontsize=10)

l = 0.8
b = 0.11
w = 0.015
h = 0.3


rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 

cb = plt.colorbar(im, cax=cbar_ax,ticks=np.arange(5)+1)
plt.savefig('img/decision_boundary.svg',bbox_inches='tight',dpi=600) 





















