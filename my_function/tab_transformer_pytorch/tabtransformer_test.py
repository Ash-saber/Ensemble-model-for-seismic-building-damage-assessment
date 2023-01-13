# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:55:56 2022

@author: Steve
"""
import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

import torch
import torch.nn as nn
from my_function.tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset #Dataset的包
import torch.optim as optim
import numpy as np
import pandas as pd
data_x = np.load('x_smp.npy')
data_y = np.load('y_smp.npy')-1
def onehot_encode(y):
    y = y.reshape(-1)
    y = y.astype('int')
    y_max = np.max(y)
    y_encode = np.zeros((len(y),y_max+1)).astype('int')
    for i in range(len(y)):
        y_encode[i,y[i]] = 1
    return y_encode
data_y = onehot_encode(data_y)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_org = data_x
data_org = pd.DataFrame(data_org)

x_dim = data_org.nunique(axis = 0, dropna = True)
x_dim = np.array(x_dim)
cat_idxs = np.arange(len(x_dim))[x_dim<=8]
cat_dims = x_dim[cat_idxs]
con_idxs = np.arange(len(x_dim))[x_dim>8]

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(data_x,data_y,
                                                    test_size = 0.3,
                                                    random_state = 0)

train_mean = np.mean((x_train),axis=0).reshape(-1)
train_std = np.std((x_train),axis=0).reshape(-1)
continuous_mean_std = np.array([train_mean[con_idxs],train_std[con_idxs]]).astype(np.float32).T

class MyData(Dataset): #我定义的这个类
    def __init__(self,x_smp,y_smp):
        self.x_smp = x_smp
        self.y_smp = y_smp
    def __getitem__(self, idx): 
    # 改写__getitem__(self,item)函数，最后得到图像，标签
        x = self.x_smp[idx,:]
        #获取标签（这里简单写了aligned与original）
        y = self.y_smp[idx,:]
        return x,y
    def __len__(self):
        return len(self.x_smp)

train_data = MyData(x_train,y_train)
trainloader = DataLoader(train_data, batch_size=512, shuffle=True,num_workers=0)

model = TabTransformer(
    categories = tuple(cat_dims),      # tuple containing the number of unique values within each category
    num_continuous = len(con_idxs),                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 5,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = torch.tensor(continuous_mean_std) # (optional) - normalize the continuous values before layer norm
)

data = next(iter(trainloader))

x_categ = data[0][:,cat_idxs].int()     # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont = data[0][:,con_idxs].float()                # assume continuous values are already normalized individually

pred = model(x_categ, x_cont) # (1, 1)




