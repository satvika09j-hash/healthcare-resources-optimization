import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv("Healthcare Resource optimization - DATA (3).csv")
X = data[['BEDS AVAILABLE','DOCTORS AVAILABLE','EQUIPMENT AVAILABLE']]
y = data['PATIENTS']
model = RandomForestRegressor()
model.fit(X,y)
st.title("Healthcare Resource Optimization Engine")
st.write("Predict hospital patient demand and optimize resources")
beds = st.number_input("Beds Available")
doctors = st.number_input("Doctors Available")
equipment = st.number_input("Equipment Available")
if st.button("Predict Demand"):
    prediction = model.predict([[beds,doctors,equipment]])
    st.write("Predicted Patients:", round(prediction[0]))
    if prediction[0] > beds:
        st.error("⚠ Resource Shortage Detected")
    else:
        st.success("Resources are sufficient")