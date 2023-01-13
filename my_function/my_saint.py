# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:05:39 2022

@author: Steve
"""
import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

import torch
from torch import nn
from my_function.saint_main.models import SAINT

import numpy as np
import pandas as pd
import pickle

from my_function.saint_main.data_openml import data_prep_openml,task_dset_ids,DataSetCatCon


import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from my_function.saint_main.utils import count_parameters, classification_scores, mean_sq_error
from my_function.saint_main.augmentations import embed_data_mask
from my_function.saint_main.augmentations import add_noise

from my_function.saint_main.my_function_saint.para_define import *

from sklearn.model_selection import train_test_split


def add_mask(x):
    mask = np.ones(x.shape).astype('int')
    x_dic = {'data':x,'mask':mask}
    return x_dic


class saint_model():
    def __init__(self, multiclass = True, epochs=8):
        self.opt = para_define()
        if multiclass == False:
            self.opt.task = 'regression'
            self.opt.dtask = 'reg'
        else:
            self.opt.task = 'multiclass'
            self.opt.dtask = 'clf'
        self.modelsave_path = os.path.join(os.getcwd(),'my_function','saint_main','bestmodel')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        print(f"Device is {self.device}.")
        torch.manual_seed(self.opt.set_seed)
        os.makedirs(self.modelsave_path, exist_ok=True)
        self.opt.epochs=epochs



    def fit(self,x_smp,y_smp):
        if self.opt.task == 'multiclass':
            y_smp = y_smp.reshape(-1,1)
        modelsave_path = self.modelsave_path
        opt = self.opt
        device = self.device
        data_org = x_smp
        data_org = pd.DataFrame(data_org)
        
        x_dim = data_org.nunique(axis = 0, dropna = True)
        x_dim = np.array(x_dim)
        cat_idxs = np.arange(len(x_dim))[x_dim<=8]
        cat_dims = x_dim[cat_idxs]
        con_idxs = np.arange(len(x_dim))[x_dim>8]
        
        self.cat_idxs = cat_idxs
        x_train, x_valid, y_train, y_valid = train_test_split(x_smp,y_smp,
                                                            test_size = 0.3,
                                                            random_state = 0)
        
        train_mean = np.mean((x_train),axis=0).reshape(-1)
        train_std = np.std((x_train),axis=0).reshape(-1)
        continuous_mean_std = np.array([train_mean[con_idxs],train_std[con_idxs]]).astype(np.float32) 
        self.continuous_mean_std = continuous_mean_std
        
        x_train, x_valid = add_mask(x_train), add_mask(x_valid)
        y_train, y_valid = add_mask(y_train), add_mask(y_valid)
        
        
        train_ds = DataSetCatCon(x_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=0)
        
        valid_ds = DataSetCatCon(x_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)
        
        if opt.dtask=='reg':
            cat_dims = np.append(np.array([5 for i in range(5)]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
            #cat_dims = np.array(cat_dims).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
        else:
            cat_dims = np.append(np.array(cat_dims),np.array(5)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.    

        model = SAINT(
        categories = tuple(cat_dims), 
        num_continuous = len(con_idxs),                
        dim = opt.embedding_size,                           
        dim_out = 1,                       
        depth = opt.transformer_depth,                       
        heads = opt.attention_heads,                         
        attn_dropout = opt.attention_dropout,             
        ff_dropout = opt.ff_dropout,                  
        mlp_hidden_mults = (4, 2),       
        cont_embeddings = opt.cont_embeddings,
        attentiontype = opt.attentiontype,
        final_mlp_style = opt.final_mlp_style,
        y_dim = 5
        )        
        if self.opt.dtask=='clf':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.MSELoss().to(device)
        model.to(device)
        vision_dset = opt.vision_dset
        if opt.pretrain:
            from pretraining import SAINT_pretrain
            model = SAINT_pretrain(model, cat_idxs,x_train,y_train, continuous_mean_std, opt,device)
        
        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
        best_valid_auroc = 0
        best_valid_accuracy = 0
        best_valid_rmse = 100000
        
        
        
        print('Training begins now.')
        for epoch in range(opt.epochs):
            model.train()
            running_loss = 0.0
            #i = 0;epoch = 0; data = next(iter(trainloader))
            for i, data in enumerate(trainloader,0):
                optimizer.zero_grad()
                # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                #y_gts = y_gts.to(torch.int64)
                # We are converting the data to embeddings in the next step
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
                reps = model.transformer(x_categ_enc, x_cont_enc)
                # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:,0,:]
                
                y_outs = model.mlpfory(y_reps)
                y_dim = y_outs.shape[1]
                if opt.task == 'regression':
                    loss_list =[]
                    for loss_idx in range(y_dim):
                        loss_list.append(criterion(y_outs[:,loss_idx].reshape(-1,1),y_gts[:,loss_idx].reshape(-1,1)))
                    loss = sum(loss_list)/y_dim
                else:
                    loss = criterion(y_outs,y_gts.squeeze().long()) 
                loss.backward()
                optimizer.step()
                if opt.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()
            # print(running_loss)
            #print(f'epoch:{epoch}    loss:{running_loss}')
            if opt.active_log:
                wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
                'loss': loss.item()
                })
            if epoch%1==0:
                    model.eval()
                    with torch.no_grad():
                        if opt.task in ['binary','multiclass']:
                            valid_accuracy, valid_auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                            #test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)
                            train_accuracy, train_auroc = classification_scores(model, trainloader, device, opt.task,vision_dset)
                            print('[EPOCH %d] TRAIN ACCURACY: %.3f VALID ACCURACY: %.3f' %
                                (epoch + 1, train_accuracy, valid_accuracy))                    
        
        
                            if opt.task =='multiclass':
                                if valid_accuracy > best_valid_accuracy:
                                    best_valid_accuracy = valid_accuracy
                                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                            else:
                                if accuracy > best_valid_accuracy:
                                    best_valid_accuracy = accuracy
                                # if auroc > best_valid_auroc:
                                #     best_valid_auroc = auroc
                                    best_test_auroc = test_auroc
                                    best_test_accuracy = test_accuracy               
                                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
        
                        else:
                            valid_rmse = mean_sq_error(model, validloader, device,vision_dset) 
                            train_rmse = mean_sq_error(model, trainloader, device,vision_dset) 
                            print('[EPOCH %d] train RMSE: %.5f VALID RMSE: %.5f' %
                                (epoch + 1, train_rmse, valid_rmse))
                            if valid_rmse < best_valid_rmse:
                                best_valid_rmse = valid_rmse
                                torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    model.train()
        model.load_state_dict(torch.load('%s/bestmodel.pth' % (modelsave_path)))
        self.model = model
        
    def predict(self,x_smp):
        cat_idxs = self.cat_idxs
        continuous_mean_std = self.continuous_mean_std
        opt = self.opt
        y_smp = np.zeros((len(x_smp),5))
        x_smp = add_mask(x_smp)
        y_smp = add_mask(y_smp)
        smp_ds = DataSetCatCon(x_smp, y_smp, cat_idxs,opt.dtask, continuous_mean_std)
        smploader = DataLoader(smp_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)
        model = self.model
        device = self.device
        vision_dset = opt.vision_dset
        with torch.no_grad():
            model.eval()
            y_pred = torch.empty(0).to(device)
            for i, data in enumerate(smploader, 0):
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:,0,:]
                y_outs = model.mlpfory(y_reps)
                y_pred = torch.cat([y_pred,y_outs],dim=0)
            y_pred = y_pred.cpu().numpy()
        return y_pred
    def predict_proba(self,x_smp):
        fun=nn.Softmax(dim=1)
        cat_idxs = self.cat_idxs
        continuous_mean_std = self.continuous_mean_std
        opt = self.opt
        y_smp = np.zeros((len(x_smp),1))
        x_smp = add_mask(x_smp)
        y_smp = add_mask(y_smp)
        smp_ds = DataSetCatCon(x_smp, y_smp, cat_idxs,opt.dtask, continuous_mean_std)
        smploader = DataLoader(smp_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)
        model = self.model
        device = self.device
        vision_dset = opt.vision_dset
        with torch.no_grad():
            model.eval()
            y_pred = torch.empty(0).to(device)
            for i, data in enumerate(smploader, 0):
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:,0,:]
                y_outs = model.mlpfory(y_reps)
                y_pred = torch.cat([y_pred,y_outs],dim=0)
            y_pred = fun(y_pred)
            y_pred = y_pred.cpu().numpy()
        return y_pred
        
        
        
        
        
        
        
        
        
        
        
        