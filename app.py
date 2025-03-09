import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def load_model():
    model=joblib.load("model.pkl")
    scaler=joblib.load("scaler.pkl")
    return model, scaler

def detect_anomalies(df, model, scaler):
    df_scaled=pd.DataFrame(scaler.transform(df), columns=df.columns)
    df_scaled["Anomaly"]=model.predict(df_scaled)
    # Calculate anomaly scores using the Isolation Forest model (model)
    df_scaled["Anomaly_Score"] = model.decision_function(df_scaled.drop(columns=["Anomaly"]))
    df["Anomaly"]=df_scaled["Anomaly"]
    df["Anomaly_Score"]=df_scaled["Anomaly_Score"]
    return df

st.title("IoT Sensor Anomaly Detection")

uploaded_file=st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    df=df.drop(columns=["Time", "Air Quality"], errors='ignore')
    
    model, scaler=load_model()
    df_anomalies=detect_anomalies(df, model, scaler)
    
    st.subheader("Anomaly Detection Results")
    
    # Display sample anomaly scores
    st.subheader("Sample Anomaly Scores")
    st.write(df_anomalies[["Anomaly", "Anomaly_Score"]].head(10))
    
    st.subheader("Anomaly Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(df_anomalies["Temperature"], df_anomalies["Humidity"], c=df_anomalies["Anomaly"], cmap="coolwarm", alpha=0.7)
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.title("Anomaly Detection Visualization")
    st.pyplot(plt)