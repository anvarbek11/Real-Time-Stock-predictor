import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from flask import Flask, render_template, request
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Constants - Hardcoded for optimization
PREDICTION_DAYS = 30  # Fixed to 30 days for performance
TIME_STEP = 60  # Fixed time step for model input
DATA_YEARS = 3  # Reduced from 5 to 3 years for lighter data load

# Load model efficiently at startup
model = load_model('stock_price_model.h5')
model.make_predict_function()  # For faster predictions


@lru_cache(maxsize=4)  # Cache last 4 stock downloads
def get_stock_data(stock_symbol):
    """Fetch and cache stock data with optimized date range"""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
    df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    return preprocess_data(df)


def preprocess_data(df):
    """Clean and prepare the DataFrame with only essential columns"""
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    df = df[['Date', 'Close']]  # Only keep Close price for predictions
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def prepare_data(df):
    """Optimized data preparation with fixed time step"""
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1).astype('float32'))

    # Efficient array creation using sliding window
    X = np.lib.stride_tricks.sliding_window_view(scaled_close[:-1], TIME_STEP, axis=0)
    y = scaled_close[TIME_STEP:, 0]
    X = X.reshape(X.shape[0], TIME_STEP, 1)

    return X, y, scaler


def predict_future(model, data, scaler):
    """Optimized 30-day prediction function"""
    last_data = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    future_preds = np.zeros(PREDICTION_DAYS, dtype='float32')  # Pre-allocated array

    for i in range(PREDICTION_DAYS):
        next_pred = model.predict(last_data, verbose=0)[0, 0]
        future_preds[i] = next_pred
        last_data = np.roll(last_data, -1, axis=1)  # More efficient than np.append
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(future_preds.reshape(-1, 1))


def create_plot(x1, y1, name1, color1, x2=None, y2=None, name2=None, color2=None, title=""):
    """Efficient plot creation function"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, name=name1, line=dict(color=color1)))
    if x2 is not None and y2 is not None:
        fig.add_trace(go.Scatter(x=x2, y=y2, name=name2, line=dict(color=color2)))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')  # Use CDN for JS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'TSLA').upper()

        try:
            # Get cached or fresh data
            df = get_stock_data(stock_symbol)

            # Prepare data
            X, y, scaler = prepare_data(df)

            # Make predictions
            y_pred = model.predict(X, verbose=0)
            y_pred = scaler.inverse_transform(y_pred)

            # 30-day future prediction
            future_prices = predict_future(model, scaler.transform(df['Close'].values.reshape(-1, 1)), scaler)
            future_dates = pd.date_range(start=df.index[-1], periods=PREDICTION_DAYS + 1)[1:]

            # Create optimized plots
            plot_main = create_plot(
                x1=df.index, y1=df['Close'], name1='Actual', color1='blue',
                x2=df.index[TIME_STEP + 1:], y2=y_pred[:, 0], name2='Predicted', color2='orange',
                title=f'{stock_symbol} Price Prediction'
            )

            plot_future = create_plot(
                x1=df.index, y1=df['Close'], name1='Historical', color1='blue',
                x2=future_dates, y2=future_prices[:, 0], name2='30-Day Forecast', color2='green',
                title=f'{stock_symbol} 30-Day Forecast'
            )

            return render_template('index.html',
                                   plot_main=plot_main,
                                   plot_future=plot_future,
                                   stock_symbol=stock_symbol,
                                   last_price=df['Close'].iloc[-1],
                                   last_date=df.index[-1].strftime('%Y-%m-%d'),
                                   prediction_days=PREDICTION_DAYS
                                   )

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)  # Debug=False for production