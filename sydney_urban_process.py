
# coding: utf-8

# In[14]:

## open_csv => input : 一個csv檔案的路徑 , return : 一個讀成 np array的csv檔案

## get_xyz_from_csv => input : 一個csv檔案的路徑 , return : 只有xyz部分的 np array

## get_eigv_from_xyz_metrix = > input : xyz 的 nparray , return : x_1, x_2, x_3, x_4


import pandas as pd
import numpy as np
from numpy import linalg as LA

def open_csv(csv_file_path):
    csv_file = pd.read_csv(csv_file_path, header=None)
    return csv_file
    
def get_xyz_from_csv(csv_file_path):
    csv_file = pd.read_csv(csv_file_path, header=None)
    csv_file = np.array(csv_file)
    
    matrix = np.array(csv_file)
    
    point_num = matrix.shape[0]
    
    xyz_metrix = np.zeros((point_num, 3))
    
    for i in range(point_num):
        for k in range(3):
            xyz_metrix[i][k] = matrix[i][k+3]
    
    return xyz_metrix

def get_eigv_from_xyz_metrix(input_metrix):
    input_metrix = input_metrix.T
    
    max_of_z = input_metrix[2].max()
    min_of_z = input_metrix[2].min()

    x_1 = max_of_z - min_of_z
    
    cov_metrix = np.cov(input_metrix)
    eig = LA.eig(cov_metrix)
    
    eig_v = eig[0]
    
    lamda_1 = eig_v[0]
    lamda_2 = eig_v[1]
    lamda_3 = eig_v[2]
    
    x_2 = lamda_3/(lamda_1*lamda_2)
    x_3 = lamda_2/lamda_3
    x_4 = (lamda_1*lamda_3)/(lamda_2*lamda_2)
    
    return eig_v,np.array([x_1,x_2,x_3,x_4])

