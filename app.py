import streamlit as st
import joblib
import numpy as np

st.title('House Price Prediction')
st.divider()

st.write('This app uses machine learning for predicting hous eprice with given features of the house. For using this app you can enter inputs from this UI and then use predict button')

st.divider()

bedrooms = st.number_input('Number of bedrooms',min_value=0,value=0)
bathrooms = st.number_input('Number of bathrooms',min_value=0,value=0)
livingarea = st.number_input('Living area',min_value=0,value=2000)
condition = st.number_input('Condition',min_value=0,value=3)
numberofschools = st.number_input('Number of schools near by',min_value=0,value=0)

st.divider()

model = joblib.load('model.pkl')

x = [[bedrooms,bathrooms,livingarea,condition,numberofschools]]

predictButton = st.button('Predict!')

if predictButton:
    st.balloons()
    x_array = np.array(x)
    prediction = model.predict(x_array)[0]
    st.write(f'Price prediction is {prediction}')
else:
    st.write('Please use predict button after entering values')

