import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Title
st.title("📦 Smart Parcel AI System")
st.write("Predict Damage Risk and Estimated Delivery Time for your parcels")

# Sidebar Inputs
st.sidebar.header("Enter Parcel Details")
parcel_type = st.sidebar.selectbox("Parcel Type", ["Fragile", "Non-Fragile"])
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=50.0, value=10.0)
distance = st.sidebar.number_input("Distance (km)", min_value=1.0, max_value=5000.0, value=500.0)
delivery_speed = st.sidebar.selectbox("Delivery Speed", ["Standard", "Express"])
weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy"])
handling_score = st.sidebar.slider("Handling Score", 1, 10, 8)
stops = st.sidebar.slider("Number of Stops", 1, 10, 2)

# Generate a small model (simulate same as Jupyter)
np.random.seed(42)
num_samples = 1000
data = pd.DataFrame({
    'parcel_type': np.random.choice(['Fragile', 'Non-Fragile'], num_samples),
    'weight': np.random.uniform(0.5, 30, num_samples),
    'distance': np.random.uniform(5, 2000, num_samples),
    'delivery_speed': np.random.choice(['Standard', 'Express'], num_samples),
    'weather': np.random.choice(['Clear', 'Rainy', 'Stormy'], num_samples),
    'handling_score': np.random.randint(1, 11, num_samples),
    'stops': np.random.randint(1, 6, num_samples)
})

risk_prob = (
    (data['parcel_type'] == 'Fragile').astype(int)*0.3 +
    (data['weight']/30)*0.2 +
    (data['stops']/5)*0.2 +
    (data['weather'] == 'Stormy').astype(int)*0.3
)
data['damage_risk'] = np.where(np.random.rand(num_samples) < risk_prob, 1, 0)

base_time = data['distance']/60
speed_factor = np.where(data['delivery_speed'] == 'Express', 0.8, 1.2)
stop_factor = data['stops']*0.3
data['delivery_time'] = base_time * speed_factor + stop_factor + np.random.normal(0, 1, num_samples)

features = ['parcel_type','weight','distance','delivery_speed','weather','handling_score','stops']
X = data[features]
y_risk = data['damage_risk']
y_time = data['delivery_time']

categorical = ['parcel_type','delivery_speed','weather']
numerical = ['weight','distance','handling_score','stops']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical),
    ('num', StandardScaler(), numerical)
])

risk_model = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
time_model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

risk_model.fit(X, y_risk)
time_model.fit(X, y_time)

# Predict new input
new_data = pd.DataFrame({
    'parcel_type': [parcel_type],
    'weight': [weight],
    'distance': [distance],
    'delivery_speed': [delivery_speed],
    'weather': [weather],
    'handling_score': [handling_score],
    'stops': [stops]
})

risk_pred = risk_model.predict(new_data)[0]
time_pred = time_model.predict(new_data)[0]

# Display results
st.subheader("Prediction Results:")
st.write(f"Damage Risk: **{'High' if risk_pred==1 else 'Low'}**")
st.write(f"Estimated Delivery Time: **{time_pred:.2f} hours**")
