#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[4]:


#Streamlit Install
get_ipython().system('pip install streamlit')
get_ipython().system('pip install PIL')


# In[7]:


df=sns.load_dataset('iris')


# In[8]:


df.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder as LE
le=LE()
df.species=le.fit_transform(df.species)
df.species


# In[13]:


X=df.drop('species',axis=1)
Y=df.species
X.columns=['sl','sw','pl','pw']

# In[19]:


from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test=tt(X,Y,test_size=0.2,random_state=3)


# In[36]:


#model building using LR
from sklearn.ensemble import RandomForestClassifier as LR


# In[37]:


lr=LR()


# In[38]:


model=lr.fit(x_train,y_train)


# In[39]:


pred=model.predict(x_test)


# In[40]:

from sklearn.metrics import accuracy_score as ac
ac(y_test,pred)


# In[41]:


from sklearn.metrics import classification_report as cr


# In[42]:


print(cr(y_test,pred))


# In[44]:


import pickle 


# In[45]:


pickle.dump(model,open('model.pkl','wb'))


# In[ ]:




