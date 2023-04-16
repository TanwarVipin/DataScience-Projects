#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st


# In[29]:


LR=pickle.load(open('model1.sav','rb'))


# In[30]:


KNR=pickle.load(open('model2.sav','rb'))


# In[31]:


SVR=pickle.load(open('model3.sav','rb'))


# In[32]:


DR=pickle.load(open('model4.sav','rb'))


# In[33]:


def LR1(x):
    st.write(x)
    pred=LR.predict(x)
    return pred
def KNR1(x):
    pred=KNR.predict(x)
    return pred
def SVR1(x):
    pred=SVR.predict(x)
    return pred
def DR1(x):
    pred=DR.predict(x)
    return pred


# In[35]:


def main():
    st.markdown(
    """
    <style>
    .reportview-container {
        background: url("url_goes_here")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True)

    st.title("CAR WEIGHT PREDICTIONS")
    HP=st.number_input('HP',key=1,min_value=0,max_value=120,step=10)
    MPG=st.number_input('MPG',key=2,min_value=0,max_value=120,step=10)
    VOL=st.number_input('VOL',key=3,min_value=0,max_value=120,step=10)
    SP=st.number_input('SP',key=4,min_value=0,max_value=120,step=10)
    variable=np.array([[HP,MPG,VOL,SP]])
    Options=['LR','KNR','SVR','DR']
    y=st.sidebar.selectbox('Choose Options',Options)
    if st.button('Predict'):
        result=''
        if y=='LR':
            result=LR1(variable)
            st.success(f'The predicted Weight is {result}')
        elif y=='KNR':
            result=KNR1(variable)
            st.success(f'The predicted Weight is {result}')
        elif y=='SVR':
            result=SVR1(variable)
            st.success(f'The predicted Weight is {result}')
        elif y=='DR':
            result=DR1(variable)
            st.success(f'The predicted Weight is {result}')
        
            


# In[36]:


if __name__=='__main__':
    main()


# In[ ]:




