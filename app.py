# app.py
import streamlit as st
import pandas as pd
import pickle
import os

st.title("📊 Customer Churn Prediction")

# --- Load model file safely ---
MODEL_FILE = "customer_churn_model.pkl"
ENCODERS_FILE = "encoders.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

if not os.path.exists(ENCODERS_FILE):
    st.error(f"Encoders file not found: {ENCODERS_FILE}")
    st.stop()

# Load model_data (might be dict) and encoders
with open(MODEL_FILE, "rb") as f:
    model_data = pickle.load(f)

# If user saved dict like {"model": rfc, "features_names": [...]}, handle it
if isinstance(model_data, dict) and "model" in model_data:
    model = model_data["model"]
    feature_names = model_data.get("features_names", None)
else:
    # assume they directly saved the model object
    model = model_data
    feature_names = None

with open(ENCODERS_FILE, "rb") as f:
    encoders = pickle.load(f)

# Show loaded info (optional)
st.write("Loaded model type:", type(model).__name__)
if feature_names:
    st.write("Model expects features (count):", len(feature_names))

# Build input UI using encoder classes for categorical fields
# Use the exact categorical columns that were saved in encoders
categorical_cols = list(encoders.keys())

st.header("Input customer details")

# Use columns to make it compact
cols = st.columns(3)

user_input = {}

# For each categorical col, show selectbox with safe options from encoder.classes_
for i, col in enumerate(categorical_cols):
    label = col  # you can prettify this if you want
    options = list(encoders[col].classes_)
    # place across columns
    c = cols[i % len(cols)]
    user_input[col] = c.selectbox(label, options)

# Numeric fields expected by your model based on notebook:
numeric_fields = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is 0/1
user_input["SeniorCitizen"] = cols[0].selectbox("SeniorCitizen", [0, 1])
user_input["tenure"] = cols[1].number_input("tenure (months)", min_value=0, max_value=100, value=12)
user_input["MonthlyCharges"] = cols[2].number_input("MonthlyCharges", min_value=0.0, value=70.0)
# TotalCharges can be larger
user_input["TotalCharges"] = st.number_input("TotalCharges", min_value=0.0, value=100.0, step=0.1)

# When user clicks predict
if st.button("Predict Churn"):
    try:
        # Make DataFrame using feature_names if present, else try to infer order
        if feature_names:
            # Build dictionary with all feature_names. If some not provided, raise helpful error.
            missing = [f for f in feature_names if f not in user_input]
            if missing:
                st.error(f"Missing input(s) for feature(s): {missing}")
                st.stop()
            input_df = pd.DataFrame([{fn: user_input[fn] for fn in feature_names}])
        else:
            # feature_names not present — rely on order: categorical_cols + numeric_fields
            cols_order = categorical_cols + numeric_fields
            input_df = pd.DataFrame([{k: user_input[k] for k in cols_order}])

        # Encode categorical columns using saved encoders
        for col, encoder in encoders.items():
            # LabelEncoder expects array-like of original labels
            input_df[col] = encoder.transform(input_df[col])

        # Ensure input_df columns order matches model's training features
        if feature_names:
            input_df = input_df[feature_names]

        # Prediction
        if hasattr(model, "predict"):
            pred = model.predict(input_df)[0]
            # try predict_proba if available for probability
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0].max()  # top class prob
            if pred == 1:
                st.error(f"🚨 Prediction: Churn (prob={prob:.3f})" if prob is not None else "🚨 Prediction: Churn")
            else:
                st.success(f"✅ Prediction: No Churn (prob={prob:.3f})" if prob is not None else "✅ Prediction: No Churn")
        else:
            st.error("Loaded model object has no predict() method. Check that you loaded the correct object from pickle.")
    except Exception as e:
        st.exception(e)
