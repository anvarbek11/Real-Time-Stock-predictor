# 📊 Real-Time Stock Predictor

This project is a **real-time stock price prediction web app** powered by a trained LSTM deep learning model. It allows users to input any stock symbol (e.g., TSLA, AAPL, MSFT) and receive predictions, future forecasts, and technical indicators through a modern interactive interface.
 
📈 **Model:** Trained LSTM (Keras)  
📦 **Frontend:** Gradio + Plotly  
📡 **Data Source:** Yahoo Finance (`yfinance`)

---

## 🔧 Features

- 📅 Fetches last **3 years** of historical stock data
- 🤖 Predicts stock prices using a **pre-trained LSTM model**
- 📉 30-day **future forecast**
- 📊 Technical indicators: **50-day & 200-day Simple Moving Averages**
- ⚡ Interactive UI built with **Gradio + Plotly**
- 💾 One-click deploy to Hugging Face Spaces

---

## 🖥️ Demo

![1](https://github.com/user-attachments/assets/1d48dbd0-c15f-494c-bd89-4d9d35108432)


---

## 🧠 Model Info

- Architecture: LSTM with sliding window
- Input: Last 60 days of closing prices
- Output: Next 30 days of predicted prices
- Training: Done offline using Keras, saved as `stock_price_model.h5`

---

