
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:30:48 2019
# theta = np.reshape(t,len(t),1)
@author: sowrabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

alpha=0.07

dataset=pd.read_csv('Salary_Data.csv')
x =np.array( dataset.iloc[:, :-1].values)
y = np.array(dataset.iloc[:, 1].values)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

m=np.size(y_train)
t = np.linalg.inv(np.dot(np.matrix.transpose(x),x))
u = np.dot(t,np.matrix.transpose(x))
theta = np.dot(u,y)

yPred=theta*x
plt.scatter(x_train,y_train,color = "m")
plt.plot(x,yPred,color="g")
plt.xlabel('age')
plt.ylabel('salary')
plt.show()
