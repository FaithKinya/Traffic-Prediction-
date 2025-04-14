import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_dashboard():
    df = pd.read_csv("data/traffic_data.csv")
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.bar_chart(df['Weather_Condition'].value_counts())

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="DayOfWeek", data=df, ax=ax)
        st.pyplot(fig)
