import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Title
st.title("📦 Smart Parcel AI System")
st.write("Predict Damage Risk, Estimated Delivery Time & Cost for your parcels")

# Sidebar Inputs
st.sidebar.header("Enter Parcel Details")
parcel_type = st.sidebar.selectbox("Parcel Type", ["Fragile", "Non-Fragile"])
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=50.0, value=10.0)
distance = st.sidebar.number_input("Distance (km)", min_value=1.0, max_value=5000.0, value=500.0)
delivery_speed = st.sidebar.selectbox("Delivery Speed", ["Standard", "Express"])
weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy"])
handling_score = st.sidebar.slider("Handling Score", 1, 10, 8)
stops = st.sidebar.slider("Number of Stops", 1, 10, 2)

# Generate synthetic training data
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

# Risk & time generation logic
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

# Features
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

# Models
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

# Predict single input
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
risk_proba = risk_model.predict_proba(new_data)[0][1]
time_pred = time_model.predict(new_data)[0]
time_low = time_pred - 2
time_high = time_pred + 2

# Cost estimation formula
base_cost = 50
cost = base_cost + (weight*5) + (distance*0.2)
if delivery_speed == "Express":
    cost *= 1.5
if parcel_type == "Fragile":
    cost *= 1.2

# Display results
st.subheader("📊 Prediction Results")
st.write(f"Damage Risk: **{'High 🔴' if risk_pred==1 else 'Low 🟢'}**")
st.write(f"Risk Probability: **{risk_proba*100:.1f}%**")
st.progress(min(int(risk_proba*100),100))  # progress bar for risk
st.write(f"Estimated Delivery Time: **{time_pred:.2f} hours** (Range: {time_low:.1f}–{time_high:.1f} hrs)")
st.write(f"Estimated Delivery Cost: **₹{cost:.2f}**")

# --- Visualization: Risk Probability Gauge ---
st.subheader("📈 Risk Probability Gauge")
fig, ax = plt.subplots()
ax.pie([risk_proba, 1-risk_proba], labels=[f"Risk {risk_proba*100:.1f}%", f"Safe {100-risk_proba*100:.1f}%"],
       autopct='%1.1f%%', colors=["red","green"], startangle=90)
ax.set_aspect('equal')
st.pyplot(fig)

# --- Visualization: Feature Importance ---
st.subheader("🔍 Feature Importance in Risk Prediction")
feature_names = list(risk_model.named_steps['preprocess'].transformers_[0][1].get_feature_names_out(categorical)) + numerical
importances = risk_model.named_steps['classifier'].feature_importances_

fig, ax = plt.subplots()
ax.barh(feature_names, importances)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

# Multiple parcel upload
st.subheader("📥 Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV with parcel details", type=["csv"])
if uploaded_file is not None:
    bulk_data = pd.read_csv(uploaded_file)
    bulk_risk = risk_model.predict(bulk_data)
    bulk_risk_proba = risk_model.predict_proba(bulk_data)[:,1]
    bulk_time = time_model.predict(bulk_data)

    # Add predictions
    bulk_data['Damage Risk'] = np.where(bulk_risk==1, "High", "Low")
    bulk_data['Risk Probability (%)'] = (bulk_risk_proba*100).round(2)
    bulk_data['Estimated Time (hrs)'] = bulk_time.round(2)

    # Cost for bulk
    bulk_data['Estimated Cost (₹)'] = (
        50 + (bulk_data['weight']*5) + (bulk_data['distance']*0.2)
    ) * np.where(bulk_data['delivery_speed']=="Express", 1.5, 1.0) * \
      np.where(bulk_data['parcel_type']=="Fragile", 1.2, 1.0)

    st.dataframe(bulk_data)

    # Download predictions
    csv = bulk_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", data=csv, file_name="parcel_predictions.csv", mime="text/csv")
