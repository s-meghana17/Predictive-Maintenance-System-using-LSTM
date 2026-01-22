import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page title
# --------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance System")
st.title("Predictive Maintenance System")

# --------------------------------------------------
# Load model and scaler (IMPORTANT: compile=False)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(
        "models/lstm_model.h5",
        compile=False   # 🔥 fixes keras.metrics.mse error
    )
    scaler = joblib.load("models/scaler.save")
    return model, scaler

model, scaler = load_artifacts()

st.success("Model and scaler loaded successfully ✅")

# --------------------------------------------------
# Load dataset (CMAPSS example)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/CMAPSSData/train_FD001.txt",
        sep=" ",
        header=None
    )
    df.dropna(axis=1, inplace=True)
    return df

data = load_data()

st.write("### Sample Data")
st.dataframe(data.head())

# --------------------------------------------------
# User input (example sensor values)
# --------------------------------------------------
st.write("### Enter Sensor Values")
st.write("### Enter Sensor Values (18 sensors)")

# Create sensor names once
sensor_names = [f"s{i}" for i in range(1, 19)]

sensor_values = []

# Create inputs
for name in sensor_names:
    val = st.number_input(f"Sensor {name}", value=0.0)
    sensor_values.append(val)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Remaining Useful Life"):
    input_array = np.array(sensor_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # reshape for LSTM → (samples, timesteps, features)
    input_scaled = input_scaled.reshape(
        (input_scaled.shape[0], 1, input_scaled.shape[1])
    )

    prediction = model.predict(input_scaled)
    rul = prediction[0][0]

    st.success(f"Predicted Remaining Useful Life (RUL): **{rul:.2f} cycles**")
