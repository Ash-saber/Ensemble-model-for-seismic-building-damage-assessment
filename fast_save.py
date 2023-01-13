# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:13:46 2022

@author: Steve
"""
#creat npy document for data2.xlsx
import os
path = 'D:\Program project\python project\pytorch\working6-ensemble'
os.chdir(path)
import numpy as np
import pandas as pd
data_org = pd.read_excel(io = 'data2.xlsx',sheet_name = 'sequence',header = 0,
                     engine='openpyxl')



data_x = np.array(data_org.iloc[:,0:21])
data_y = np.array(data_org.iloc[:,21])

np.save('x_allsmp.npy',data_x)
np.save('y_allsmp.npy',data_y)