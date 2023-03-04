# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:35:18 2022

@author: Steve
"""

import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

import torch
from torch import nn
from models import SAINT

import numpy as np
import pandas as pd
import pickle



from my_function.saint-main.data_openml import data_prep_openml,task_dset_ids,DataSetCatCon


import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from my_function.saint_main.utils import count_parameters, classification_scores, mean_sq_error
from my_function.saint_main.augmentations import embed_data_mask
from my_function.saint_main.augmentations import add_noise

from my_function.saint_main.my_function_saint.para_define import *
opt = para_define()

modelsave_path = os.path.join(os.getcwd(),'./bestmodel','for_earthquake')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")
torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)



##########################使用70w个完整数据
#data_org = pd.read_excel(io = 'my_data/data2.xlsx',sheet_name = 'sequence',header = 0,
#                     engine='openpyxl')

data_org = np.concatenate((np.load('x_smp.npy'),np.load('y_smp.npy').reshape(-1,1)),axis=1)
data_org = pd.DataFrame(data_org)


x_dim = data_org.nunique(axis = 0, dropna = True)
x_dim = np.array(x_dim)[0:21]
cat_idxs = np.arange(len(x_dim))[x_dim<=8]
cat_dims = x_dim[cat_idxs]
con_idxs = np.arange(len(x_dim))[x_dim>8]


data_x = np.array(data_org.iloc[:,0:21])
data_y = np.array(data_org.iloc[:,21]).astype('int').reshape(-1,1)-1




from sklearn.model_selection import train_test_split

x_smp, _, y_smp, _ = train_test_split(data_x,data_y,
                                        test_size = 0.01,
                                        random_state = 0)
#np.save('x_smp.npy',x_smp)
#np.save('y_smp',y_smp)

x_train, x_test, y_train, y_test = train_test_split(x_smp,y_smp,
                                                    test_size = 0.3,
                                                    random_state = 0)
x_valid, x_test, y_valid, y_test = train_test_split(x_test,y_test,
                                                    test_size = 0.5,
                                                    random_state = 0)

train_mean = np.mean((x_train),axis=0).reshape(-1)
train_std = np.std((x_train),axis=0).reshape(-1)
continuous_mean_std = np.array([train_mean[con_idxs],train_std[con_idxs]]).astype(np.float32) 

def add_mask(x):
    mask = np.ones(x.shape).astype('int')
    x_dic = {'data':x,'mask':mask}
    return x_dic


x_train, x_valid, x_test = add_mask(x_train), add_mask(x_valid), add_mask(x_test)
y_train, y_valid, y_test = add_mask(y_train), add_mask(y_valid), add_mask(y_test)

train_ds = DataSetCatCon(x_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=0)

valid_ds = DataSetCatCon(x_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)

test_ds = DataSetCatCon(x_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)


cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

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
y_dim = 1
)

vision_dset = opt.vision_dset
criterion = nn.CrossEntropyLoss().to(device)
model.to(device)

if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000

#i = 0
#data = next(iter(trainloader))

opt.epochs=100
print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
        y_gts = y_gts.to(torch.int64)
        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs,y_gts) 
        else:
            loss = criterion(y_outs,y_gts.squeeze()) 
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
                    accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                    test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)
                    train_accuracy, train_auroc = classification_scores(model, trainloader, device, opt.task,vision_dset)
                    print('[EPOCH %d] TRAIN ACCURACY: %.3f' %
                        (epoch + 1, train_accuracy))                    
                    print('[EPOCH %d] VALID ACCURACY: %.3f' %
                        (epoch + 1, accuracy))
                    print('[EPOCH %d] TEST ACCURACY: %.3f' %
                        (epoch + 1, test_accuracy))

                    if opt.task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
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
                    test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                    print('[EPOCH %d] VALID RMSE: %.3f' %
                        (epoch + 1, valid_rmse ))
                    print('[EPOCH %d] TEST RMSE: %.3f' %
                        (epoch + 1, test_rmse ))
                    if opt.active_log:
                        wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()
                




















