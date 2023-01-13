# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:44:15 2023

@author: Steve
"""
import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

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
with open ('model_save/stack_model_c_3.pkl','rb') as f:
    stack_model = pickle.load(f)
    

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')

y_true = y_valid
yp_stack = stack_model.predict(x_valid).argmax(1)
yp_boost = boost_model.predict(x_valid).argmax(1)
c_s = confusion_matrix(y_true,yp_stack)
c_b = confusion_matrix(y_true,yp_boost)

acc_s = acc_cal(yp_stack,y_valid)
acc_b = acc_cal(yp_boost,y_valid)




def confusion_matrix_plot(c_mat,acc,fig_name):
    from matplotlib.colors import LinearSegmentedColormap
    my_cmap = LinearSegmentedColormap.from_list('mycmap', ['#CDDFF1','#2676B7','#08306B'])
    my_cmap2 = LinearSegmentedColormap.from_list('mycmap', ['#BB1419','#BB1419'])
    p = precision_cal(c_mat)
    r = recall_cal(c_mat)
    acc = np.array([acc])
    
    fig = plt.figure(dpi=600,figsize=(5,5))
    gs = plt.GridSpec(32,32)   
    plt.rc('font',family='Times New Roman')
    plt.rcParams['font.size']=13.5
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    ax1 = fig.add_subplot(gs[0:25,0:25])  
    ax2 = fig.add_subplot(gs[0:25,27:])
    ax3 = fig.add_subplot(gs[27:,0:25])
    ax4 = fig.add_subplot(gs[27:,27:])
    
    ax1.matshow(c_mat, interpolation='nearest', cmap=my_cmap)
    ax1.text(2,-1,'Predicted DG',ha='center',weight='bold')
    ax1.set_ylabel('True DG',weight='bold')
    dg_text = [f'DG{i+1}' for i in range(5)]
    ax1.set_yticklabels(['']+dg_text) 
    ax1.set_xticklabels(['']+dg_text) 
    for i in range(len(c_mat)):
        for j in range(len(c_mat)):
            ax1.annotate(c_mat[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center',
                         color = 'white'if i==4 and j==4 else 'black')
    ax1.tick_params(size=0)
    
    ax2.matshow(r.reshape(-1,1), interpolation='nearest', cmap='Reds')
    for i in range(len(c_mat)):
        ax2.annotate(f'{round(r[i]*100,1)}%', xy=(0,i), horizontalalignment='center', verticalalignment='center',
                     color ='black' if i!=4 else 'white')
    ax2.text(0,-0.7,'Recall',ha='center',weight='bold')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params(size=0)
    
    ax3.matshow(p.reshape(1,-1), interpolation='nearest', cmap='Reds')
    ax3.text(-1,0.5,'Precision',ha='center',rotation=90,weight='bold')
    for i in range(len(c_mat)):
        ax3.annotate(f'{round(p[i]*100,1)}%', xy=(i,0), horizontalalignment='center', verticalalignment='center',
                     color = 'black' if i!=4 else 'white')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.tick_params(size=0)
    
    ax4.matshow(acc.reshape(1,1), interpolation='nearest', cmap=my_cmap2)
    ax4.annotate(f'{round(acc[0]*100,1)}%', xy=(0,0), horizontalalignment='center', verticalalignment='center',
                 color = 'black')
    ax4.text(0,-0.6,'Accuracy',ha='center',weight='bold')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.tick_params(size=0)  

    plt.savefig(f'img/{fig_name}.svg',bbox_inches='tight',dpi=600)
    
confusion_matrix_plot(c_s,acc_s,'stack_confusion_matrix')    
confusion_matrix_plot(c_b,acc_b,'boost_confusion_matrix')    


from sklearn.metrics import roc_curve, auc

# 计算
def roc_plot(yt,yp_proba,fig_name):

    fpr_list = []; tpr_list =[]; auc_list =[]
    for i in range(5):
        fpr, tpr, thread = roc_curve(yt==i, yp_proba[:,i])
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)
    
    color_list = ['#4C72B0','#DD8452','#55A868','#C44E52','#C44E52']
    color_list2 = ['#FF0000','#FF9F00','#0101C9','#007E00','purple']
    line_list=['-','-.',':','--','-']
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111) 
    plt.rcParams['font.size']=12
    for i in range(5):
        ax.plot(fpr_list[i],tpr_list[i],linestyle=line_list[i],lw=2,color=color_list2[i],label=f'DG{i+1} (AUC = %0.2f)' % auc_list[i])
    ax.plot([0, 1], [0, 1], color='#DDDDDD', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    #plt.rcParams.update({'font.size':10})
    plt.legend(loc="lower right",)
    plt.savefig(f'img/{fig_name}.svg',bbox_inches='tight',dpi=600)

yp_stack_proba = stack_model.predict(x_valid)
yp_boost_proba = boost_model.predict(x_valid)
roc_plot(y_valid,yp_stack_proba,'stack_roc')
roc_plot(y_valid,yp_boost_proba,'boost_roc')

yp_proba_list=[yp_stack_proba,yp_boost_proba]



with open ('model_save/stack_model_c.pkl','rb') as f:
    stack_model_6 = pickle.load(f)

    

for i in range(6):
    yp_temp_proba = stack_model_6.model_list[i].predict_proba(x_valid)
    yp_proba_list.append(yp_temp_proba)

sub_name_list = ['Stacking','Boosting','TabNet','XGBoost','SAINT','Light GBM','Tab-transformer','Random Forest']    
    
    
    
def multi_rocplot(yt,yp_proba_list,sub_name_list):
    fig,ax=plt.subplots(4,2,figsize=(11, 15),dpi=600)   
    fig.tight_layout(h_pad=3)
    ax = ax.ravel()
    num = list(str('abcdefgh'))
    for j in range(8):
        yp_proba = yp_proba_list[j]
        fpr_list = []; tpr_list =[]; auc_list =[]
        for i in range(5):
            fpr, tpr, thread = roc_curve(yt==i, yp_proba[:,i])
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
        
        color_list = ['#4C72B0','#DD8452','#55A868','#C44E52','#C44E52']
        color_list2 = ['#B20000','#FF9F00','purple','#007E00','navy']
        line_list=['-',':','-','--','-.']
        plt.rcParams['font.size']=10
        plt.rc('font',family='Times New Roman')
        for i in range(5):
            ax[j].plot(fpr_list[i],tpr_list[i],linestyle=line_list[i],lw=2,color=color_list2[i],label=f'DG{i+1} (AUC = {round(auc_list[i],2)})')
        ax[j].legend(frameon=False)
        #plt.legend(loc="lower right")
        ax[j].plot([0, 1], [0, 1], color='#DDDDDD', lw=1.5, linestyle='--')
        ax[j].set_xlim([0.0, 1.0])
        ax[j].set_ylim([0.0, 1.05])
        plt.rcParams['font.size']=12
        plt.rc('font',family='Times New Roman')
        ax[j].set_xlabel('False Positive Rate')
        ax[j].set_ylabel('True Positive Rate')
        #plt.rcParams.update({'font.size':10})
        ax[j].set_title(f'({num[j]}) {sub_name_list[j]}',y=-0.25,fontsize=14)
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['right'].set_visible(False)
        
    plt.savefig('img/roc_curves_for_all.svg',bbox_inches='tight',dpi=600)    


multi_rocplot(y_valid, yp_proba_list, sub_name_list)    
yp_list = [yp_proba_list[i].argmax(1) for i in range(8)]
avg_recall = [np.mean(recall_cal(confusion_matrix(y_valid,yp_list[i]))) for i in range(8)]
avg_precision = [np.mean(precision_cal(confusion_matrix(y_valid,yp_list[i]))) for i in range(8)]

weight_list = np.array(percent_cal(data_y))

weighted_recall = [np.sum(recall_cal(confusion_matrix(y_valid,yp_list[i]))*weight_list) for i in range(8)]
weighted_precision = [np.sum(precision_cal(confusion_matrix(y_valid,yp_list[i]))*weight_list) for i in range(8)]

acc = [acc_cal(y_valid,yp_list[i]) for i in range(8)]

a = np.concatenate((np.array(avg_recall).reshape(-1,1),np.array(avg_precision).reshape(-1,1),
                np.array(weighted_recall).reshape(-1,1),np.array(weighted_precision).reshape(-1,1),
                np.array(acc).reshape(-1,1)),axis=1)
a = a*100




