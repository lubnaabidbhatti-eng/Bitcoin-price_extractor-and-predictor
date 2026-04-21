# crypto_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tabulate import tabulate

st.set_page_config(layout="wide")
st.title("Crypto Dashboard - BTC/USDT")

# -----------------------
# BINANCE CONFIG
# -----------------------
exchange = ccxt.binance({'enableRateLimit': True})
symbol = "BTC/USDT"

# -----------------------
# SIDEBAR OPTIONS
# -----------------------
st.sidebar.header("Dashboard Controls")

# Live price settings
live_candles = st.sidebar.slider("Number of candles for live chart", 5, 100, 20)
interval_live = st.sidebar.selectbox("Live Data Interval", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"])

# Historical chart settings
hist_start = st.sidebar.date_input("Historical start date", date.today()-timedelta(days=180))
hist_end = st.sidebar.date_input("Historical end date", date.today())
interval_hist = st.sidebar.selectbox("Historical Interval", ["1d","1w","1M"])

# LSTM prediction settings
predict_days = st.sidebar.slider("Future prediction days", 5, 30, 10)

# Email report option
send_email = st.sidebar.checkbox("Send Daily Email Report")

# -----------------------
# FUNCTION: FETCH OHLCV
# -----------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, interval, limit=None, since=None):
    data = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, since=since)
    df = pd.DataFrame(data, columns=["Timestamp","Open","High","Low","Close","Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = df[col].astype(float)
    return df

# -----------------------
# FUNCTION: FETCH CURRENT PRICE + 24h High/Low + Volume + Time
# -----------------------
@st.cache_data(ttl=30)
def get_current_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    high = ticker['high']
    low = ticker['low']
    vol = ticker['quoteVolume']          # Total traded volume in quote currency
    timestamp = pd.to_datetime(ticker['timestamp'], unit='ms')
    return price, high, low, vol, timestamp

# -----------------------
# FUNCTION: LSTM PREDICTION
# -----------------------
def lstm_predict(closing_prices, days_ahead):
    # Ensure correct shape (2D)
    closing_prices = np.array(closing_prices).reshape(-1, 1)

    if len(closing_prices) < 100:
        raise ValueError("Not enough data. Need at least 100 data points.")
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(closing_prices)
    x_data, y_data = [], []
    base_days = 100
    for i in range(base_days, len(scaled_data)):
        x_data.append(scaled_data[i-base_days:i])
        y_data.append(scaled_data[i])
    x_data, y_data = np.array(x_data), np.array(y_data)
    split = int(len(x_data)*0.9)
    x_train, y_train = x_data[:split], y_data[:split]
    x_test, y_test = x_data[split:], y_data[split:]
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)
    
    # Predict future
    last_100 = scaled_data[-100:].reshape(1,100,1)
    future = []
    for _ in range(days_ahead):
        next_val = model.predict(last_100, verbose=0)
        future.append(next_val)
        last_100 = np.append(last_100[:,1:,:], next_val.reshape(1,1,1), axis=1)
    future = np.array(future).reshape(-1,1)
    future_prices = scaler.inverse_transform(future)
    return future_prices.flatten()

# -----------------------
# PANEL 1: LIVE PRICE
# -----------------------
st.subheader("Current Price & 24h High/Low")
live_df = fetch_ohlcv(symbol, interval_live, live_candles)
price, high, low, vol, time = get_current_price(symbol)

fig_live = go.Figure()
fig_live.add_trace(go.Candlestick(
    x=live_df["Timestamp"], open=live_df["Open"], high=live_df["High"],
    low=live_df["Low"], close=live_df["Close"], name="Price"
))
fig_live.add_trace(go.Scatter(
    x=live_df["Timestamp"], y=[high]*len(live_df), mode="lines", name="24h High", line=dict(color="green")
))
fig_live.add_trace(go.Scatter(
    x=live_df["Timestamp"], y=[low]*len(live_df), mode="lines", name="24h Low", line=dict(color="red")
))
fig_live.update_layout(title=f"BTC/USDT Live Price - Current: {price:.2f} USD", height=400)
st.plotly_chart(fig_live, use_container_width=True)

# -----------------------
# PANEL 2: HISTORICAL CHART
# -----------------------
st.subheader("Historical Price Chart")
since_ms = int(datetime.combine(hist_start, datetime.min.time()).timestamp()*1000)
hist_df = fetch_ohlcv(symbol, interval_hist, since=since_ms)
fig_hist = go.Figure()
fig_hist.add_trace(go.Candlestick(
    x=hist_df["Timestamp"], open=hist_df["Open"], high=hist_df["High"],
    low=hist_df["Low"], close=hist_df["Close"], name="Historical"
))
fig_hist.update_layout(title=f"BTC/USDT Historical Chart ({interval_hist})", height=400)
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------
# PANEL 3: MOVING AVERAGES
# -----------------------
st.subheader("Moving Averages")
hist_df["MA_50"] = hist_df["Close"].rolling(50).mean()
hist_df["MA_200"] = hist_df["Close"].rolling(200).mean()
fig_ma = go.Figure()
fig_ma.add_trace(go.Candlestick(
    x=hist_df["Timestamp"], open=hist_df["Open"], high=hist_df["High"],
    low=hist_df["Low"], close=hist_df["Close"], name="Price"
))
fig_ma.add_trace(go.Scatter(x=hist_df["Timestamp"], y=hist_df["MA_50"], name="MA 50", line=dict(color="blue")))
fig_ma.add_trace(go.Scatter(x=hist_df["Timestamp"], y=hist_df["MA_200"], name="MA 200", line=dict(color="red")))
fig_ma.update_layout(title="BTC/USDT Price with Moving Averages", height=400)
st.plotly_chart(fig_ma, use_container_width=True)

# -----------------------
# PANEL 4: LSTM PREDICTION
# -----------------------
st.subheader("LSTM Future Price Prediction")
closing_prices = hist_df["Close"].dropna().values.reshape(-1,1)
future_prices = lstm_predict(closing_prices, predict_days)
future_dates = [hist_df["Timestamp"].iloc[-1] + timedelta(days=i+1) for i in range(predict_days)]
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=hist_df["Timestamp"], y=closing_prices.flatten(), mode="lines", name="Actual Price"))
fig_pred.add_trace(go.Scatter(x=future_dates, y=future_prices, mode="lines+markers", name="Predicted Price", line=dict(color="purple")))
fig_pred.update_layout(title=f"BTC/USDT LSTM Prediction for {predict_days} Days", height=400)
st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------
# PANEL 5: SEND EMAIL REPORT
# -----------------------
from dotenv import load_dotenv
import os
if send_email:
    st.subheader("Sending Email Report")
    
    # Prepare report table
    table = pd.DataFrame({
        "Metric": ["Current Price","24h High","24h Low","Volume","Time"],
        "Value": [price, high, low, vol, time]
    })
    
    subject = f"BTC/USDT Daily Report - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    body = f"Hello,\n\nHere is the BTC report:\n\n{tabulate(table, headers='keys', tablefmt='grid')}\n\nRegards,\nCrypto Dashboard"
    
    def send_mail(subject, body):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_mail = os.getenv("SENDER_MAIL")
        email_password = os.getenv("EMAIL_PASSWORD")
        receiver_mail =os.getenv("RECEIVER_MAIL")
     
        message = MIMEMultipart()
        message['From'] = sender_mail
        message['To'] = receiver_mail
        message['Subject'] = subject
        message.attach(MIMEText(body,'plain'))
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_mail, email_password)
                server.sendmail(sender_mail, receiver_mail, message.as_string())
            st.success("Email sent successfully!")
        except Exception as e:
            st.error(f"Email sending failed: {e}")
    
    send_mail(subject, body)