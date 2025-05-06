import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from flask import Flask, render_template, request
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
import gc
# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load pre-trained model
model = load_model('stock_price_model.h5')
model.build((None, 60, 1))  # Explicitly build with expected input shape

# Warm up the model
model.predict(np.zeros((1, 60, 1)))


def preprocess_data(df):
    """Process yfinance data to match training format"""
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'Date'})
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_stock_data(stock_symbol, years=2):  # Reduced from 5 to 2 years
    """Fetch and preprocess stock data"""
    df = yf.download(stock_symbol, period=f"{years}y")  # More efficient than date range
    return preprocess_data(df)


def prepare_data(df, time_step=60):
    """Prepare data for LSTM prediction"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])  # More efficient than reshape

    # Vectorized window creation
    X = np.lib.stride_tricks.sliding_window_view(scaled_data.flatten(), time_step)[:-1]
    y = scaled_data[time_step:]
    return X.reshape(-1, time_step, 1), y, scaler


def predict_future(model, data, scaler, time_step=60, future_days=30):
    """Generate future predictions (max 60 days)"""
    future_days = min(future_days, 60)  # Enforce maximum 60 days prediction
    last_data = data[-time_step:].reshape(1, time_step, 1)

    future_prices = []
    for _ in range(future_days):
        next_pred = model.predict(last_data, verbose=0)[0, 0]  # Disable logging
        future_prices.append(next_pred)
        last_data = np.roll(last_data, -1)
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'TSLA').upper()
        future_days = min(int(request.form.get('future_days', 30)), 60)  # Cap at 60 days

        try:
            # Get and prepare data
            df = get_stock_data(stock_symbol)
            X, y, scaler = prepare_data(df)

            # Make predictions
            y_pred = model.predict(X, batch_size=32, verbose=0)  # Batch prediction
            y_pred = scaler.inverse_transform(y_pred)

            # Future prediction
            future_prices = predict_future(model,
                                           scaler.transform(df[['Close']]),
                                           scaler,
                                           future_days=future_days)
            future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1)[1:]

            # Create plots (simplified layout)
            def create_plot(title):
                fig = go.Figure()
                fig.update_layout(
                    title=title,
                    template='plotly_dark',
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                return fig

            fig_main = create_plot(f'{stock_symbol} Price Prediction')
            fig_main.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual'))
            fig_main.add_trace(go.Scatter(x=df.index[60:], y=y_pred[:, 0], name='Predicted'))

            fig_future = create_plot(f'{future_days}-Day Forecast')
            fig_future.add_trace(go.Scatter(x=df.index, y=df['Close'], name='History'))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_prices[:, 0], name='Forecast'))

            # Technical indicators
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()

            fig_tech = create_plot('Technical Indicators')
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA'))
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA'))

            # Manual memory cleanup
            del X
            gc.collect()

            return render_template('index.html',
                                   plot_main=fig_main.to_html(full_html=False),
                                   plot_future=fig_future.to_html(full_html=False),
                                   plot_tech=fig_tech.to_html(full_html=False),
                                   stock_symbol=stock_symbol,
                                   future_days=future_days,
                                   last_price=round(df['Close'].iloc[-1], 2),
                                   last_date=df.index[-1].strftime('%Y-%m-%d'))

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)  # Disable debug for production