import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from flask import Flask, render_template, request
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load pre-trained model
model = load_model('stock_price_model.h5')
model.build((None, 60, 1))  # Explicitly build with expected input shape


def preprocess_data(df):
    """Process yfinance data to match training format"""
    # Flatten multi-index columns
    df.columns = [col[0] for col in df.columns]

    # Reset index and rename columns to match training
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]

    # Convert and set date index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_stock_data(stock_symbol, years=5):
    """Fetch and preprocess stock data"""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * years)
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return preprocess_data(df)


def prepare_data(df, time_step=60):
    """Prepare data for LSTM prediction"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Create dataset for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler


def predict_future(model, data, scaler, time_step=60, future_days=30):
    """Generate future predictions"""
    last_data = data[-time_step:]
    last_data = last_data.reshape(1, time_step, 1)

    future_predictions = []
    for _ in range(future_days):
        # Ensure input shape matches training shape
        next_pred = model.predict(last_data)
        future_predictions.append(next_pred[0, 0])
        last_data = np.append(last_data[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'TSLA').upper()
        future_days = int(request.form.get('future_days', 30))

        try:
            # Get and prepare data
            df = get_stock_data(stock_symbol)
            X, y, scaler = prepare_data(df)

            # Verify input shape
            print(f"Input shape to model: {X.shape}")  # Should be (n_samples, 60, 1)

            # Make predictions
            y_pred = model.predict(X)
            y_pred = scaler.inverse_transform(y_pred)
            y_actual = scaler.inverse_transform(y.reshape(-1, 1))

            # Future prediction
            future_prices = predict_future(model,
                                           scaler.transform(df['Close'].values.reshape(-1, 1)),
                                           scaler,
                                           future_days=future_days)
            future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1)[1:]

            # Create plots
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                          name='Actual Price', line=dict(color='blue')))
            fig_main.add_trace(go.Scatter(x=df.index[61:], y=y_pred[:, 0],
                                          name='Predicted Price', line=dict(color='orange')))
            fig_main.update_layout(title=f'{stock_symbol} Stock Price Prediction',
                                   template='plotly_dark')
            plot_main = fig_main.to_html(full_html=False)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                            name='Historical Price', line=dict(color='blue')))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_prices[:, 0],
                                            name='Future Prediction', line=dict(color='green')))
            fig_future.update_layout(title=f'{stock_symbol} {future_days}-Day Forecast',
                                     template='plotly_dark')
            plot_future = fig_future.to_html(full_html=False)

            # Technical indicators
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()

            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                          name='Price', line=dict(color='blue')))
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                                          name='50-Day SMA', line=dict(color='orange')))
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['SMA_200'],
                                          name='200-Day SMA', line=dict(color='red')))
            fig_tech.update_layout(title=f'{stock_symbol} Technical Indicators',
                                   template='plotly_dark')
            plot_tech = fig_tech.to_html(full_html=False)

            return render_template('index.html',
                                   plot_main=plot_main,
                                   plot_future=plot_future,
                                   plot_tech=plot_tech,
                                   stock_symbol=stock_symbol,
                                   future_days=future_days,
                                   last_price=df['Close'].iloc[-1],
                                   last_date=df.index[-1].strftime('%Y-%m-%d'))

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)