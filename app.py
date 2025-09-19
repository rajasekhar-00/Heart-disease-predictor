# Heart Disease Predictor - Streamlit App with All Advanced Features

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import base64
import csv
from datetime import datetime

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset for accuracy
data = pd.read_csv("heart_disease.csv")
X = data.drop("condition", axis=1)
y = data["condition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Page Config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")


# Sidebar
st.sidebar.title("🩺 Heart Disease Predictor")
st.sidebar.markdown(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

# Input Form
st.title("💓 Predict Your Heart Disease Risk")

with st.form("prediction_form"):
    age = st.slider("🧓 Age", 20, 80, 50)
    sex = st.radio("🚻 Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("🔥 Chest Pain Type", [0, 1, 2, 3])
    trtbps = st.slider("💉 Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("🧬 Cholesterol", 100, 600, 200)
    fbs = st.radio("🍭 Fasting Blood Sugar > 120", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("📉 Resting ECG", [0, 1, 2])
    thalachh = st.slider("❤️ Max Heart Rate Achieved", 70, 210, 150)
    exng = st.radio("🏃 Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("📉 Oldpeak", 0.0, 6.0, 1.0, step=0.1)
    slp = st.selectbox("📈 Slope", [0, 1, 2])
    ca = st.selectbox("🧪 Number of Major Vessels (CA)", [0, 1, 2, 3, 4])
    thall = st.selectbox("💊 Thalassemia", [0, 1, 2, 3])

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, ca, thall]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Log the input
    with open("user_data_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now()] + list(input_data[0]))

    # Show Result
    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease")
    else:
        st.success(f"✅ Low Risk of Heart Disease")

    st.markdown(f"**Risk Score**: {probability * 100:.2f}%")
    st.progress(probability)

    # Show input bar chart
    st.subheader("📊 Input Summary")
    labels = ['Age', 'Sex', 'CP', 'BP', 'Chol', 'FBS', 'ECG', 'HR', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal']
    fig, ax = plt.subplots()
    ax.barh(labels, input_data[0])
    st.pyplot(fig)

    # Recommendation
    st.subheader("📌 Recommendation")
    if prediction[0] == 1:
        st.write("🧑‍⚕️ Please consult a cardiologist as soon as possible.")
    else:
        st.write("💪 Keep maintaining a healthy lifestyle!")

    # Download Report
    report = f"""
    Heart Disease Report
    ----------------------
    Age: {age}
    Risk Score: {probability * 100:.2f}%
    Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="heart_report.txt">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)
  
