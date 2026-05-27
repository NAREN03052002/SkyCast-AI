import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from weather_api import get_weather

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SkyCast AI",
    page_icon="🌦️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    with open("model/rainfall_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/weather_AUS.csv")
    return df

df = load_data()

# Remove unwanted column if present
df_display = df.drop(columns=["Unnamed: 0"], errors="ignore")

# -----------------------------
# HEADER
# -----------------------------
st.title("🌦️ SkyCast AI")
st.markdown("## Rainfall Prediction & Weather Analytics System")

st.markdown("---")


# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🌍 Weather Location")

city = st.sidebar.text_input(
    "Enter City",
    value="Mangalore"
)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🔮 Rainfall Prediction")

if st.sidebar.button("🌧 Fetch Weather & Predict"):

    weather = get_weather(city)

    if weather is None:
        st.error("City not found")
        st.stop()

    temp = weather["temp"]
    humidity = weather["humidity"]
    pressure = weather["pressure"]
    wind_speed = weather["wind_speed"]

    st.success(f"Weather fetched for {city}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{temp} °C")
    col2.metric("Humidity", f"{humidity}%")
    col3.metric("Pressure", f"{pressure} hPa")
    col4.metric("Wind Speed", f"{wind_speed} m/s")

    sample = pd.DataFrame({
        "Location":["Sydney"],
        "MinTemp":[temp-2],
        "MaxTemp":[temp+2],
        "Rainfall":[0],
        "Evaporation":[5.0],
        "Sunshine":[8.0],
        "WindGustDir":["N"],
        "WindGustSpeed":[30],
        "WindDir9am":["N"],
        "WindDir3pm":["NE"],
        "WindSpeed9am":[wind_speed],
        "WindSpeed3pm":[wind_speed],
        "Humidity9am":[humidity],
        "Humidity3pm":[humidity],
        "Pressure9am":[pressure],
        "Pressure3pm":[pressure],
        "Cloud9am":[4],
        "Cloud3pm":[4],
        "Temp9am":[temp],
        "Temp3pm":[temp],
        "RainToday":["No"]
    })

    prediction = model.predict(sample)[0]

    if prediction == "Yes":
        st.error("🌧 High Chance of Rain Tomorrow")
    else:
        st.success("☀ No Rain Expected Tomorrow")

# -----------------------------
# DATASET PREVIEW
# -----------------------------
with st.expander("📊 View Dataset Sample"):

    st.dataframe(
        df_display.head(20),
        use_container_width=True
    )

st.markdown("---")

# -----------------------------
# PROJECT DETAILS
# -----------------------------
st.subheader("📘 Project Information")

st.write("""
**Project Name:** SkyCast AI

**Objective:** Predict rainfall using machine learning and historical weather data.

**Dataset:** Rain in Australia Dataset

**Algorithm:** Random Forest Classifier

**Accuracy:** 84.69%

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit
""")