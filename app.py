# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:16:02 2023

@author: user
"""

import pickle
clf=pickle.load(open("C:/Users/user/Desktop/New folder/model.sav",'rb'))
import streamlit as st
def welcome():
    return 'Welcome All'
def prediction(sl,sw,pl,pw):
    pred=clf.predict([[sl,sw,pl,pw]])
    return pred
def main():
    st.title('Iris Flower Prediction')
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Iris Flower Classifier ML App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sl=st.text_input('sepal_length','')
    sw=st.text_input('sepal_width','')
    pl=st.text_input('petal_length','')
    pw=st.text_input('petal_width','')
    l=[sl,sw,pl,pw]
    result=''
    if st.button('Predict'):
        result=prediction(sl,sw,pl,pw)

    if result==0:
        st.success(f'The flower Type is Virginica')
    elif result==1:
        st.success(f'The flower Type is Setosa')
    elif result==2:
        st.success(f'The Flower Type is Versicolor')
    
if __name__=='__main__':
    main()

