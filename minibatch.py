# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:34:42 2018

@author: adi
"""
# cost function

def cost(x,y,theta):
    j=np.power((x.dot(theta.T)-y),2)
    j=(1/(2*len(x)))*np.sum(j)
    
    return j
# importing library

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#importing data

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

 # splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1) 

# feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
y_train=sc_X.fit_transform(y_train)
y_test=sc_X.fit_transform(y_test)
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.fit_transform(x_test)

 

x_train=np.append(arr=np.ones((24,1)).astype( int),values=x_train,axis=1)
x_test=np.append(arr=np.ones((6,1)).astype(int),values=x_test,axis=1)
theta=np.array([[1.0,1.0]])

# mini batch gradient descent 

al=0.01 
for i in range(1000): 
    for j in range(0,24,4):
      theta=theta-(al/4)*(x_train[j:j+4].dot(theta.T)-y_train[j:j+4]).T.dot(x_train[j:j+4])      

# predicted outcome
                            
y_pred=theta[0][0]+theta[0][1]*x_test[:,1]

