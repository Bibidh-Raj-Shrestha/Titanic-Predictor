import streamlit as st
import pickle
import random
import pandas as pd

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -----------------------
# Header
# -----------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🚢 Titanic Survival Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict survival based on passenger details. "
    "Try different scenarios and explore survival chances!</p>",
    unsafe_allow_html=True
)
st.divider()

# -----------------------
# Sidebar inputs for single passenger
# -----------------------
st.sidebar.header("Single Passenger Details")
name = st.sidebar.text_input("Name")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex_input = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 0 if sex_input == "Male" else 1
age = st.sidebar.slider("Age", 1, 80, 25)

# -----------------------
# Assign realistic fare (dynamic)
# -----------------------
if pclass == 1:
    base_fare = 80 if sex == 1 else 70
elif pclass == 2:
    base_fare = 55 if sex == 1 else 45
else:
    base_fare = 20 if sex == 1 else 15

# Slight random variation for dynamic probability
fare = round(random.uniform(base_fare - 5, base_fare + 5), 2)
st.sidebar.info(f"Assigned Fare: £{fare}")

# -----------------------
# Load pre-trained model
# -----------------------
model = pickle.load(open("titanic_model.pkl", "rb"))

# -----------------------
# Single Passenger Prediction
# -----------------------
if st.sidebar.button("Predict Survival"):
    features = [[pclass, sex, age, fare]]
    prob = model.predict_proba(features)[0][1]
    
    # Add slight random "dynamic" variation
    prob_dynamic = min(max(prob + random.uniform(-0.05, 0.05), 0), 1)
    prediction = 1 if prob_dynamic >= 0.5 else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ {name or 'Passenger'} would likely SURVIVE")
    else:
        st.error(f"❌ {name or 'Passenger'} would likely NOT survive")
    
    st.metric(label="Survival Probability", value=f"{prob_dynamic*100:.2f}%")
    st.progress(int(prob_dynamic*100))

st.divider()

# -----------------------
# Multiple Passenger "What-If" Scenario
# -----------------------
st.subheader("📊 What-If Scenario: Multiple Passengers")
st.markdown("Add multiple passengers below to compare survival probabilities.")

# Number of passengers
num_passengers = st.number_input(
    "Number of passengers to simulate", min_value=1, max_value=10, value=3, step=1
)

passengers = []

# Input details for each passenger
for i in range(num_passengers):
    st.markdown(f"**Passenger {i+1} Details:**")
    pname = st.text_input(f"Name {i+1}", key=f"name{i}")
    pclass_i = st.selectbox(f"Class {i+1}", [1, 2, 3], key=f"class{i}")
    sex_i_input = st.radio(f"Sex {i+1}", ["Male", "Female"], key=f"sex{i}")
    sex_i = 0 if sex_i_input == "Male" else 1
    age_i = st.slider(f"Age {i+1}", 1, 80, 25, key=f"age{i}")

    # Assign realistic fare dynamically
    if pclass_i == 1:
        base_fare_i = 80 if sex_i == 1 else 70
    elif pclass_i == 2:
        base_fare_i = 55 if sex_i == 1 else 45
    else:
        base_fare_i = 20 if sex_i == 1 else 15

    fare_i = round(random.uniform(base_fare_i - 5, base_fare_i + 5), 2)
    passengers.append([pname, pclass_i, sex_i, age_i, fare_i])

# Predict multiple passengers
if st.button("Predict Multiple Passengers"):
    results = []
    for p in passengers:
        features = [[p[1], p[2], p[3], p[4]]]
        prob = model.predict_proba(features)[0][1]
        # Slight random variation for showcase
        prob_dynamic = min(max(prob + random.uniform(-0.05, 0.05), 0), 1)
        prediction = "Survive ✅" if prob_dynamic >= 0.5 else "Not Survive ❌"
        results.append({
            "Name": p[0] or "Passenger",
            "Probability (%)": round(prob_dynamic*100, 2),
            "Prediction": prediction
        })

    # Display results as table
    df_results = pd.DataFrame(results)
    st.table(df_results)

    # Bar chart for probabilities
    st.bar_chart(df_results.set_index("Name")["Probability (%)"])

st.divider()

# -----------------------
# Footer
# -----------------------
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Developed by You | Powered by Python, scikit-learn & Streamlit 🚀</p>",
    unsafe_allow_html=True
)
