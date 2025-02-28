import streamlit as st
import numpy as np
import joblib  

model = joblib.load("house_price_stack_model.pkl")

st.title("House Price Prediction")
st.write("Enter the details below to predict the house price.")

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
living_area = st.number_input("Living Area (sqft)", min_value=100, step=10)
lot_area = st.number_input("Lot Area (sqft)", min_value=500, step=50)
floors = st.number_input("Number of Floors", min_value=1, step=1)
waterfront = st.selectbox("Waterfront Present?", [0, 1])
views = st.number_input("Number of Views", min_value=0, step=1)
house_area = st.number_input("House Area (excluding basement) (sqft)", min_value=100, step=10)
basement_area = st.number_input("Basement Area (sqft)", min_value=0, step=10)
built_year = st.number_input("Built Year", min_value=1800, max_value=2025, step=1)
renovation_year = st.number_input("Renovation Year", min_value=0, max_value=2025, step=1)
schools_nearby = st.number_input("Number of Schools Nearby", min_value=0, step=1)
distance_airport = st.number_input("Distance from Airport (in K.M)", min_value=0.1, step=0.1)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)


if st.button("Predict Price"):
    input_features = np.array([
        bedrooms, bathrooms, living_area, lot_area, floors, waterfront, views,
        house_area, basement_area, built_year, renovation_year, schools_nearby,
        distance_airport, year, month
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = abs(model.predict(input_features)[0])
    
    st.success(f"Estimated House Price: {prediction:,.2f} INR")
