import streamlit as st
from models.model_loader import ModelHandler
from database.db_handler import DatabaseHandler
from utils import validators
from eda.eda_dashboard import show_dashboard
import pandas as pd

st.set_page_config(page_title="Smart Traffic Predictor", layout="wide")

# CSS Styling
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model_handler = ModelHandler("data/model.pkl")
db_handler = DatabaseHandler()

st.title("🚦 Smart Traffic Prediction App")

menu = st.sidebar.radio("Navigate", ["Prediction", "EDA Dashboard"])

if menu == "Prediction":
    st.subheader("🧠 Predict Traffic Condition")

    hour = st.slider("Hour of the Day", 0, 23)
    weather = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Cloudy"])
    light = st.selectbox("Traffic Light State", ["Red", "Yellow", "Green"])
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    valid_hour, hour_msg = validators.validate_hour(hour)

    if st.button("Predict"):
        if valid_hour:
            input_df = pd.DataFrame({
                'Hour': [hour],
                'Weather_Condition': [weather],
                'Traffic_Light_State': [light],
                'DayOfWeek': [day]
            })
            prediction = model_handler.predict(input_df)[0]
            st.success(f"🚗 Predicted Traffic Condition: **{prediction}**")
            db_handler.insert_input(hour, weather, light, day, prediction)
        else:
            st.error(hour_msg)

elif menu == "EDA Dashboard":
    show_dashboard()
