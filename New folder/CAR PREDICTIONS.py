#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import pandas as pd


# In[8]:


df=pd.read_csv(r"C:\Users\user\Desktop\datasets\Cars.csv")


# In[9]:


df.head()


# In[10]:


df.shape


# In[12]:


df.info()


# In[21]:


X=df.drop('WT',axis=1)
Y=df.WT


# In[22]:


from sklearn.model_selection import train_test_split as tt


# In[26]:


x_train,x_test,y_train,y_test=tt(X,Y,test_size=0.15,random_state=34)


# In[27]:


from sklearn.linear_model import LinearRegression as LR


# In[29]:


lr=LR()


# In[31]:


model1=lr.fit(x_train,y_train)


# In[33]:


model1.score(x_test,y_test)


# In[34]:


pred=model1.predict(x_test)


# In[35]:


pred


# In[37]:


from sklearn.metrics import mean_squared_error as mse


# In[39]:


np.sqrt(mse(y_test,pred))


# In[40]:


import pickle


# In[42]:


pickle.dump(model1,open('model1.sav','wb'))


# In[43]:


from sklearn.neighbors import KNeighborsRegressor as KNR


# In[68]:


knr=KNR(n_neighbors=3)


# In[69]:


model2=knr.fit(x_train,y_train)


# In[70]:


model2.score(x_test,y_test)


# In[71]:


pred2=model2.predict(x_test)


# In[72]:


pred2


# In[73]:


np.sqrt(mse(y_test,pred2))


# In[74]:


pickle.dump(model2,open('model2.sav','wb'))


# In[75]:


from sklearn.svm import SVR


# In[76]:


svr=SVR(kernel='linear')


# In[77]:


model3=svr.fit(x_train,y_train)


# In[78]:


model3.score(x_test,y_test)


# In[79]:


pred3=model3.predict(x_test)


# In[80]:


pred3


# In[81]:


np.sqrt(mse(y_test,pred3))


# In[82]:


pickle.dump(model3,open('model3.sav','wb'))


# In[83]:


from sklearn.tree import DecisionTreeRegressor as DR


# In[84]:


dr=DR(min_samples_leaf=5)


# In[85]:


model4=dr.fit(x_train,y_train)


# In[86]:


model4.score(x_test,y_test)


# In[87]:


pred4=model4.predict(x_test)


# In[88]:


pred4


# In[89]:


np.sqrt(mse(y_test,pred4))


# In[90]:


pickle.dump(model4,open('model4.sav','wb'))


# In[91]:


import os


# In[92]:


os.getcwd()


# In[111]:


x=np.array([[49,53,89,104]])


# In[112]:


x


# In[116]:


model2.predict(x)


# In[ ]:




