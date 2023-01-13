# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:05:26 2022

@author: Steve
"""

import pandas as pd
from numpy import *
def GRA_ONE(DataFrame,m=0):
    gray= DataFrame
    #读取为df格式
    #gray=(gray - gray.min()) / (gray.max() - gray.min())
    #标准化
    std=gray.iloc[:,m]#为标准要素
    ce=gray.iloc[:,0:]#为比较要素
    n=ce.shape[0]
    m=ce.shape[1]#计算行列
 
    #与标准要素比较，相减
    a=zeros([m,n])
    for i in range(m):
        for j in range(n):
            a[i,j]=abs(ce.iloc[j,i]-std[j])
    #取出矩阵中最大值与最小值
    c=amax(a)
    d=amin(a)
    #计算值
    result=zeros([m,n])
    for i in range(m):
        result[i,j]=(d+0.5*c)/(a[i,j]+0.5*c)
    #求均值，得到灰色关联值
    result2=zeros(m)
    for i in range(m):
        result2[i]=mean(result[i,:])
    RT=pd.DataFrame(result2)
    return RT
def GRA(DataFrame):
    list_columns = [str(s) for s in range(len(DataFrame.columns)) if s not in [None]]
    df_local = pd.DataFrame(columns=list_columns)
    for i in range(len(DataFrame.columns)):
        df_local.iloc[:,i] = GRA_ONE(DataFrame,m=i)[0]
    return df_local