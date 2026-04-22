import streamlit as st
from sfx import get_data, create_feature, predict_future, train_model
import matplotlib.pyplot as plt, pandas as pd

st.set_page_config(
    page_title="SignalForge-X",
    layout="wide"
)

st.title("SIGNALFORGE-X")
exchange = "binance"
coin = st.text_input("Write the Coin you want to predict eg.BTC/USDT")
tf = "5m"

@st.cache_data(ttl=120)  
def load_data(coin, tf):
    return get_data(coin, tf)

@st.cache_data
def feature(df):
    return create_feature(df)

@st.cache_resource
def ml(df, steps):
    return predict_future(df, steps)
@st.cache_resource
def ml2(df, steps):
    return train_model(df, steps)

if exchange and coin and tf:
    df = load_data(coin,tf)
    st.subheader("Sample Data")
    st.dataframe(df.tail())
    df = feature(df)
    steps = st.number_input("Enter the no of preds", min_value=1, step=1)
    if st.button("ML01"):
        if steps:
            y_f = ml(df, steps)
            st.write(y_f)
            actual = df["close"].tail(25)
            last_time = df.index[-1]
            future_index = pd.date_range(start=last_time,periods=len(y_f) + 1,freq="5min")[1:]
            pred_series = pd.Series(y_f, index=future_index)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(actual.index, actual.values, label="Actual")
            ax.plot(pred_series.index, pred_series.values, '--', label="Predicted")
            ax.legend()
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
    if st.button("ML02"):
        if steps:
            y_f = ml2(df, steps)
            st.write(y_f)
            actual = df["close"].tail(25)
            last_time = df.index[-1]
            future_index = pd.date_range(start=last_time,periods=len(y_f) + 1,freq="5min")[1:]
            pred_series = pd.Series(y_f, index=future_index)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(actual.index, actual.values, label="Actual")
            ax.plot(pred_series.index, pred_series.values, '--', label="Predicted")
            ax.legend()
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)






