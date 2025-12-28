import streamlit as st
import pickle
import random

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict survival based on passenger details")

# Load pre-trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

# User input
name = st.text_input("Name")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex_input = st.radio("Sex", ["Male", "Female"])
sex = 0 if sex_input == "Male" else 1
age = st.slider("Age", 1, 80, 25)

# Assign random fare based on class
if pclass == 1:
    fare = random.uniform(70, 100)
elif pclass == 2:
    fare = random.uniform(40, 69)
else:
    fare = random.uniform(10, 39)

st.info(f"🎟️ Randomized Fare: £{fare:.2f}")

# Predict button
if st.button("Predict Survival"):
    features = [[pclass, sex, age, fare]]
    prob = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success(f"✅ {name or 'Passenger'} would likely SURVIVE")
    else:
        st.error(f"❌ {name or 'Passenger'} would likely NOT survive")

    st.write(f"📊 Survival Probability: **{prob*100:.2f}%**")
