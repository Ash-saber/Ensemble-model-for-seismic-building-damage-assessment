# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:43:08 2022

@author: Steve
"""

import os
import numpy as np

path = 'D:\Program project\python project\pytorch\working6-ensemble'#工作路径
os.chdir(path)

import argparse

def para_define():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dset_id', type=int,default =188)
    parser.add_argument('--vision_dset', action = 'store_true')
    parser.add_argument('--task', type=str,default = 'multiclass')
    parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
    
    parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])
    
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default= 1 , type=int)
    parser.add_argument('--dset_seed', default= 5 , type=int)
    parser.add_argument('--active_log', action = 'store_true')
    
    parser.add_argument('--pretrain', action = 'store_true')
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
    parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)
    
    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)
    
    parser.add_argument('--ssl_avail_y', default= 0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])
    
    
    opt = parser.parse_args()
    opt.dtask = 'clf'
    return opt