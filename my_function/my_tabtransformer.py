# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:06:35 2022

@author: Steve
"""


import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

import torch
import torch.nn as nn
from my_function.tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset #Dataset的包
from sklearn.metrics import roc_auc_score, mean_squared_error

import numpy as np
import pandas as pd
def classification_scores(model, dloader, device,cat_idxs,con_idxs):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ = data[0][:,cat_idxs].int().to(device)    
            x_cont = data[0][:,con_idxs].float().to(device)   
            y_t = data[1].to(device)
            y_outs = model(x_categ,x_cont).argmax(1)
            y_test = torch.cat([y_test,y_t.squeeze()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc.cpu().numpy()

def mean_sq_error(model, dloader, device,cat_idxs,con_idxs):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ = data[0][:,cat_idxs].int().to(device)    
            x_cont = data[0][:,con_idxs].float().to(device)   
            y_t = data[1].to(device)
            y_outs = model(x_categ,x_cont)
            y_test = torch.cat([y_test,y_t.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse

class MyData(Dataset): #我定义的这个类
    def __init__(self,x_smp,y_smp):
        self.x_smp = x_smp
        self.y_smp = y_smp
    def __getitem__(self, idx): 
    # 改写__getitem__(self,item)函数，最后得到图像，标签
        x = self.x_smp[idx,:]
        #获取标签（这里简单写了aligned与original）
        y = self.y_smp[idx]
        return x,y
    def __len__(self):
        return len(self.x_smp)

class tab_transformer():
    def __init__(self,multiclass = True, epochs=8):
        self.modelsave_path = os.path.join(os.getcwd(),'my_function','tab_transformer_pytorch','bestmodel')
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is {self.device}.")
        os.makedirs(self.modelsave_path, exist_ok=True)
        self.epochs=epochs
        self.multiclass = multiclass


    def fit(self,x_smp,y_smp):
        modelsave_path = self.modelsave_path
        device = self.device
        data_org = x_smp
        data_org = pd.DataFrame(data_org)

        x_dim = data_org.nunique(axis = 0, dropna = True)
        x_dim = np.array(x_dim)
        cat_idxs = np.arange(len(x_dim))[x_dim<=8]
        cat_dims = x_dim[cat_idxs]
        con_idxs = np.arange(len(x_dim))[x_dim>8]
        self.cat_idxs = cat_idxs
        self.con_idxs = con_idxs
        
        from sklearn.model_selection import train_test_split
        x_train, x_valid, y_train, y_valid = train_test_split(x_smp,y_smp,
                                                            test_size = 0.3,
                                                            random_state = 0)

        train_mean = np.mean((x_train),axis=0).reshape(-1)
        train_std = np.std((x_train),axis=0).reshape(-1)
        continuous_mean_std = np.array([train_mean[con_idxs],train_std[con_idxs]]).astype(np.float32).T
        
        train_data = MyData(x_train,y_train)
        trainloader = DataLoader(train_data, batch_size=256, shuffle=True,num_workers=0)
        valid_data = MyData(x_valid,y_valid)
        validloader = DataLoader(valid_data, batch_size=256, shuffle=False,num_workers=0)
        
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
            continuous_mean_std = torch.tensor(continuous_mean_std).to(device) # (optional) - normalize the continuous values before layer norm
        )    
        if self.multiclass==True:
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(),lr=0.0001)
        best_valid_auroc = 0
        best_valid_accuracy = 0
        best_valid_rmse = 100000

        print('Training begins now.')
        for epoch in range(self.epochs):
            model.train()
            train_running_loss = 0.0
            #i = 0;epoch = 0; data = next(iter(trainloader))
            for i, data in enumerate(trainloader,0):
                optimizer.zero_grad()
                x_categ = data[0][:,cat_idxs].int().to(device)     # category values, from 0 - max number of categories, in the order as passed into the constructor above
                x_cont = data[0][:,con_idxs].float().to(device)    # assume continuous values are already normalized individually
                y_outs = model(x_categ,x_cont)
                y_dim = y_outs.shape[1]
                
                y_t = data[1].float().to(device)
                if self.multiclass==True:
                    loss = criterion(y_outs,y_t.long())
                else:  
                    loss_list =[]
                    for loss_idx in range(y_dim):
                        loss_list.append(criterion(y_outs[:,loss_idx].reshape(-1,1),y_t[:,loss_idx].reshape(-1,1)))
                    loss = sum(loss_list)/y_dim
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()
                
            if epoch%1==0:
                    model.eval()
                    with torch.no_grad():
                        if self.multiclass==True:
                            train_accuracy = classification_scores(model, trainloader, device,cat_idxs,con_idxs)
                            valid_accuracy = classification_scores(model, validloader, device,cat_idxs,con_idxs)
                            print('[EPOCH %d] TRAIN ACCURACY: %.3f VALID ACCURACY: %.3f' %
                                (epoch + 1, train_accuracy, valid_accuracy))                    
        
                            if valid_accuracy > best_valid_accuracy:
                                best_valid_accuracy = valid_accuracy
                                torch.save(model.state_dict(),'%s/trans_bestmodel.pth' % (modelsave_path))
                        else:            
                            valid_rmse = mean_sq_error(model, validloader, device,cat_idxs,con_idxs) 
                            train_rmse = mean_sq_error(model, trainloader, device,cat_idxs,con_idxs) 
                            print('[EPOCH %d] train RMSE: %.5f VALID RMSE: %.5f' %
                                (epoch + 1, train_rmse, valid_rmse))
                            if valid_rmse < best_valid_rmse:
                                best_valid_rmse = valid_rmse
                                torch.save(model.state_dict(),'%s/trans_bestmodel.pth' % (modelsave_path))
                    model.train()
        model.load_state_dict(torch.load('%s/trans_bestmodel.pth' % (modelsave_path)))
        self.model = model
        
    def predict(self,x_smp):
        device = self.device
        model = self.model
        x_smp = x_smp
        y_temp = torch.tensor(np.zeros((len(x_smp),5)))
        smp_data = MyData(x_smp,y_temp)
        smploader = DataLoader(smp_data, batch_size=512, shuffle=False,num_workers=0)
        with torch.no_grad():
            model.eval()
            y_pred = torch.empty(0).to(device)
            for i, data in enumerate(smploader, 0):
                cat_idxs = self.cat_idxs
                con_idxs = self.con_idxs
                x_categ = data[0][:,cat_idxs].int().to(device)     # category values, from 0 - max number of categories, in the order as passed into the constructor above
                x_cont = data[0][:,con_idxs].float().to(device)    # assume continuous values are already normalized individually
                y_outs = model(x_categ,x_cont)
                y_pred = torch.cat([y_pred,y_outs],dim=0)
            y_pred = y_pred.cpu().numpy()
        return y_pred
        
    def predict_proba(self,x_smp):
        fun=nn.Softmax(dim=1)
        device = self.device
        model = self.model
        x_smp = x_smp
        y_temp = torch.tensor(np.zeros((len(x_smp),1)))
        smp_data = MyData(x_smp,y_temp)
        smploader = DataLoader(smp_data, batch_size=512, shuffle=False,num_workers=0)
        with torch.no_grad():
            model.eval()
            y_pred = torch.empty(0).to(device)
            for i, data in enumerate(smploader, 0):
                cat_idxs = self.cat_idxs
                con_idxs = self.con_idxs
                x_categ = data[0][:,cat_idxs].int().to(device)     # category values, from 0 - max number of categories, in the order as passed into the constructor above
                x_cont = data[0][:,con_idxs].float().to(device)    # assume continuous values are already normalized individually
                y_outs = model(x_categ,x_cont)
                y_pred = torch.cat([y_pred,y_outs],dim=0)
            y_pred = fun(y_pred)
            y_pred = y_pred.cpu().numpy()
        return y_pred
        
        
        
        
        
        
        
        
        
        