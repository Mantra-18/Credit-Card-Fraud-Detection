import pandas as pd 
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
import shap
import joblib

model=joblib.load('paysim_xgb_optimized.pkl')
scaler=joblib.load('paysim_scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection System")
st.markdown("Enter transaction details to check if it's **Fraudulent** or **Legitimate**. You can also adjust the **decision threshold** to experiment live.")

st.sidebar.header("📋 Transaction Details")
transaction_type = st.sidebar.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'CASH_IN', 'DEBIT', 'PAYMENT'])
step = st.sidebar.slider("Step (Hour of simulation)", min_value=1, max_value=744, value=100)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", min_value=0.0, value=2000.0)
newbalanceDest = st.sidebar.number_input("New Balance (Destination)", min_value=0.0, value=3000.0)

threshold = st.sidebar.slider("🎛️ Decision Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

type_mapping = {'CASH_OUT': 0, 'TRANSFER': 1, 'CASH_IN': 2, 'DEBIT': 3, 'PAYMENT': 4}
type_encoded = type_mapping[transaction_type]

input_features = np.array([[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
input_scaled = scaler.transform(input_features)

fraud_probability = model.predict_proba(input_scaled)[:, 1][0]
prediction = int(fraud_probability >= threshold)

st.write(f"### 🔎 Fraud Probability: `{fraud_probability:.2%}`")
st.write(f"### ⚙️ Current Threshold: `{threshold:.2f}`")

if prediction == 1:
    st.error("🚨 **ALERT! This transaction is predicted to be FRAUDULENT.**")
else:
    st.success("✅ **This transaction is predicted to be LEGITIMATE.**")

if st.button(" Show Feature Importance (SHAP)"):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)
    st.write("###  Feature Contributions to Prediction")
    shap.waterfall_plot(shap_values[0])
    st.pyplot(bbox_inches="tight")

st.markdown("---")
st.markdown("Developed by **Mantra Gupta** | Trained on PaySim Dataset | Powered by XGBoost & Streamlit")
st.markdown("⚠️This application is for demonstration purposes only. Predictions are based on a trained machine learning model and may not reflect real-world outcomes.")





