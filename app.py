from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly
import plotly.graph_objects as go
import json
import sqlite3
import hashlib
import os
from functools import wraps
import time
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
app.secret_key = "your_secret_key_123"
app.config['DATABASE'] = 'stocksense.db'
app.config['PREDICTION_CACHE_DAYS'] = 3
app.config['THROTTLE_DELAY'] = 0.5
app.config['MAX_TRAINING_YEARS'] = 5

# Logging configuration
import logging
from logging.handlers import RotatingFileHandler

if not os.path.exists('logs'):
    os.makedirs('logs')
    
file_handler = RotatingFileHandler('logs/stocksense.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('StockSense startup')

# Global variable for request throttling
LAST_REQUEST_TIME = 0

# Default stock symbols
DEFAULT_STOCKS = {
    'Indian Stocks': {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'LT.NS': 'Larsen & Toubro',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'SBIN.NS': 'State Bank of India',
        'BAJFINANCE.NS': 'Bajaj Finance',
        'M&M.NS': 'Mahindra & Mahindra'
    },
    'US Stocks': {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet (Google)',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta Platforms (Facebook)',
        'NVDA': 'NVIDIA',
        'NFLX': 'Netflix',
        'KO': 'Coca-Cola',
        'JNJ': 'Johnson & Johnson',
    }
}

CURRENCY_SYMBOLS = {
    'INR': '₹',
    'USD': '$',
    'KRW': '₩',
    'JPY': '¥',
    'GBP': '£',
    'EUR': '€',
    'CNY': '¥'
}

# Database initialization
def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        db.execute("""
        CREATE TABLE IF NOT EXISTS predictions_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            predictions TEXT NOT NULL,
            company_info TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, start_date, end_date)
        );
        """)
        db.commit()

def migrate_db():
    with app.app_context():
        db = get_db()
        try:
            result = db.execute("PRAGMA table_info(predictions_cache)").fetchall()
            columns = [col[1] for col in result]
            if 'company_info' not in columns:
                db.execute("ALTER TABLE predictions_cache ADD COLUMN company_info TEXT NOT NULL DEFAULT '{}'")
            if 'high' not in columns:
                db.execute("ALTER TABLE predictions_cache ADD COLUMN high REAL")
            if 'low' not in columns:
                db.execute("ALTER TABLE predictions_cache ADD COLUMN low REAL")
            if 'volume' not in columns:
                db.execute("ALTER TABLE predictions_cache ADD COLUMN volume REAL")
            db.commit()
            app.logger.info("Database schema migrated successfully")
        except Exception as e:
            app.logger.error(f"Migration failed: {str(e)}")
            db.rollback()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize database
if not os.path.exists(app.config['DATABASE']):
    init_db()
else:
    migrate_db()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Request throttling decorator
def throttle_requests(min_interval=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global LAST_REQUEST_TIME
            elapsed = time.time() - LAST_REQUEST_TIME
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            LAST_REQUEST_TIME = time.time()
            return result
        return wrapper
    return decorator

# Context processor
@app.context_processor
def inject_defaults():
    return dict(
        default_stocks=DEFAULT_STOCKS,
        current_year=datetime.now().year
    )

# Template filters
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            return value
    return value.strftime(format)

@app.template_filter('format_number')
def format_number(value):
    try:
        if value is None:
            return "0"
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    except (ValueError, TypeError):
        return "0"
    
@app.template_filter('tojson')
def to_json(value):
    return json.dumps(value)

# Stock validation
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def is_valid_ticker(ticker):
    if not ticker or ticker.upper() == 'NONE':
        return False
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get('symbol'):
            return False
        
        hist = stock.history(period="1d")
        if hist.empty:
            return False
            
        return True
    except Exception as e:
        app.logger.warning(f"Ticker validation failed for {ticker}: {str(e)}")
        return False

# Enhanced data fetching
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_yfinance_data(ticker, years=1):
    try:
        if not ticker:
            raise ValueError("Empty ticker provided")
            
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            raise ValueError("No stock info available")
        
        period = f'{years}y' if years > 1 else '1y'
        data = yf.download(ticker, period=period, interval='1d', progress=False)
        
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")
            
        return data
        
    except Exception as e:
        app.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        raise ValueError(f"Could not fetch data for {ticker}. Please try another symbol.")

# Enhanced company info
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1mo")

        if ticker.endswith('.NS'):
            info['country'] = 'India'
            info['currency'] = 'INR'
            if 'marketCap' in info and info.get('currency') == 'USD':
                info['marketCap'] *= 75
        
        current_price = (info.get('currentPrice') or 
                        info.get('regularMarketPrice') or 
                        info.get('previousClose'))
        
        currency = info.get('currency', 'USD')
        
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'symbol': ticker,
            'sector': info.get('sector', info.get('industry', 'N/A')),
            'industry': info.get('industry', info.get('sector', 'N/A')),
            'marketCap': info.get('marketCap', info.get('totalAssets')),
            'website': info.get('website', '#'),
            'summary': info.get('longBusinessSummary', 
                             info.get('summary', 'No description available')),
            'country': info.get('country', 'N/A'),
            'currency': currency,
            'exchange': (info.get('exchangeName') or 
                        info.get('fullExchangeName') or 
                        info.get('exchange', 'N/A')),
            'employees': info.get('fullTimeEmployees'),
            'revenue': info.get('totalRevenue', info.get('revenue')),
            'dividendYield': info.get('dividendYield'),
            'peRatio': info.get('trailingPE', info.get('forwardPE')),
            'beta': info.get('beta'),
            '52WeekHigh': info.get('fiftyTwoWeekHigh'),
            '52WeekLow': info.get('fiftyTwoWeekLow'),
            'averageVolume': info.get('averageVolume', info.get('averageDailyVolume10Day')),
            'currentPrice': current_price,
            'previousClose': info.get('previousClose'),
            'openPrice': history.iloc[-1]['Open'] if not history.empty else current_price,
            'volume': history.iloc[-1]['Volume'] if not history.empty else info.get('averageVolume'),
            'currencySymbol': get_currency_symbol(currency)
        }
    except Exception as e:
        app.logger.error(f"Couldn't fetch company info for {ticker}: {str(e)}")
        return default_company_info(ticker)

def default_company_info(ticker):
    return {
        'name': ticker,
        'symbol': ticker,
        'sector': 'N/A',
        'industry': 'N/A',
        'marketCap': None,
        'website': '#',
        'summary': 'No description available',
        'country': 'N/A',
        'currency': 'USD',
        'exchange': 'N/A',
        'employees': None,
        'revenue': None,
        'dividendYield': None,
        'peRatio': None,
        'beta': None,
        '52WeekHigh': None,
        '52WeekLow': None,
        'averageVolume': None,
        'currentPrice': None,
        'previousClose': None,
        'openPrice': None,
        'volume': None,
        'currencySymbol': '$'
    }
   
def get_currency_symbol(currency_code):
    return CURRENCY_SYMBOLS.get(currency_code, '$')

# Model functions
def prepare_training_data(data, sequence_length=60, prediction_days=7):
    scaler = MinMaxScaler()
    prices = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    scaled_data = scaler.fit_transform(prices)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_days):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+prediction_days].flatten())
    return np.array(X), np.array(y), scaler

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(35)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def get_cached_prediction(ticker, start_date, end_date):
    db = get_db()
    try:
        result = db.execute(
            """SELECT predictions, company_info FROM predictions_cache 
            WHERE ticker = ? AND start_date = ? AND end_date = ? 
            AND created_at > datetime('now', ?)""",
            (ticker, start_date, end_date, f"-{app.config['PREDICTION_CACHE_DAYS']} days")
        ).fetchone()
        if result:
            return {
                'predictions': json.loads(result['predictions']),
                'company_info': json.loads(result['company_info'])
            }
    except Exception as e:
        app.logger.error(f"Error checking cache: {str(e)}")
    return None

def cache_prediction(ticker, start_date, end_date, predictions, company_info):
    db = get_db()
    try:
        db.execute(
            """INSERT OR REPLACE INTO predictions_cache 
            (ticker, start_date, end_date, predictions, company_info) 
            VALUES (?, ?, ?, ?, ?)""",
            (ticker, start_date, end_date, json.dumps(predictions), json.dumps(company_info))
        )
        db.commit()
    except Exception as e:
        app.logger.error(f"Failed to cache prediction: {str(e)}")

# Visualization functions
def create_price_chart(predictions, ticker, currency_symbol, historical_data=None):
    fig = go.Figure()
    
    if historical_data is not None and not historical_data.empty:
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data.index = pd.to_datetime(historical_data.index)
        
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Historical Close',
            line=dict(color='#9ca3af', width=1.5),
            hovertemplate='Date: %{x}<br>Close: %{y:.2f} '+currency_symbol
        ))
    
    dates = [p['date'] for p in predictions]
    fig.add_trace(go.Scatter(
        x=dates,
        y=[p['open'] for p in predictions],
        name=f'Predicted Open ({currency_symbol})',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='Date: %{x}<br>Open: %{y:.2f} '+currency_symbol
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=[p['close'] for p in predictions],
        name=f'Predicted Close ({currency_symbol})',
        line=dict(color='#ef4444', width=2),
        hovertemplate='Date: %{x}<br>Close: %{y:.2f} '+currency_symbol
    ))
    
    fig.update_layout(
        title=f"{ticker} Price History and Predictions",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        template="plotly_white",
        hovermode='x unified',
        showlegend=True
    )
    return fig

def create_moving_averages_chart(predictions, ticker, currency_symbol, historical_data=None):
    fig = go.Figure()
    
    if historical_data is not None and not historical_data.empty:
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data.index = pd.to_datetime(historical_data.index)
        
        historical_data['MA50'] = historical_data['Close'].rolling(window=50).mean()
        historical_data['MA100'] = historical_data['Close'].rolling(window=100).mean()
        
        recent_data = historical_data.tail(90)
        
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            name='Historical Close',
            line=dict(color='#9ca3af', width=1.5, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['MA50'],
            name='50-Day MA',
            line=dict(color='#10b981', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['MA100'],
            name='100-Day MA',
            line=dict(color='#f59e0b', width=2)
        ))
    
    dates = [p['date'] for p in predictions]
    fig.add_trace(go.Scatter(
        x=dates,
        y=[p['close'] for p in predictions],
        name='Predicted Close',
        line=dict(color='#ef4444', width=2)
    ))
    
    fig.update_layout(
        title=f"{ticker} Moving Averages",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        template="plotly_white",
        hovermode='x unified'
    )
    return fig

def create_warm_chart(predictions, ticker, currency_symbol):
    fig = go.Figure()
    
    dates = [p['date'] for p in predictions]
    opens = [p['open'] for p in predictions]
    closes = [p['close'] for p in predictions]
    
    changes = [(c - o) / o * 100 for o, c in zip(opens, closes)]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=closes,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False
    ))
    
    for i, (date, close, change) in enumerate(zip(dates, closes, changes)):
        color_rgb = f"rgba({255 if change < 0 else 0}, {255 if change > 0 else 0}, 0, {min(abs(change)/5, 1)})"
        
        fig.add_trace(go.Scatter(
            x=[date],
            y=[close],
            mode='markers',
            marker=dict(size=12, color=color_rgb, line=dict(width=1, color='black')),
            hovertemplate=f'Date: {date}<br>Close: {currency_symbol}{close:.2f}<br>Change: {change:.2f}%',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{ticker} Warm Chart",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        template="plotly_white"
    )
    return fig

def create_manhattan_chart(predictions, ticker, currency_symbol):
    fig = go.Figure()
    
    dates = [p['date'] for p in predictions]
    opens = [p['open'] for p in predictions]
    closes = [p['close'] for p in predictions]
    
    colors = ['green' if c >= o else 'red' for o, c in zip(opens, closes)]
    
    for i, (date, open_price, close_price, color) in enumerate(zip(dates, opens, closes, colors)):
        fig.add_trace(go.Bar(
            x=[date],
            y=[abs(close_price - open_price)],
            base=[min(open_price, close_price)],
            marker_color=color,
            name=date,
            showlegend=False,
            hovertemplate=f'Date: {date}<br>Open: {currency_symbol}{open_price:.2f}<br>Close: {currency_symbol}{close_price:.2f}'
        ))
    
    fig.update_layout(
        title=f"{ticker} Manhattan Chart",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        template="plotly_white",
        bargap=0.2
    )
    return fig

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    today = datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    return render_template('app.html',
                         current_date=today,
                         current_end_date=next_week,
                         predictions=[],
                         graphJSON=None,
                         company=None,
                         ticker=None,
                         currency_symbol='$')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form_data = request.form
    ticker = form_data.get('ticker', '').upper().strip()
    years = int(form_data.get('years', 1))
    
    if not ticker:
        flash('Please enter a stock symbol', 'error')
        return redirect(url_for('dashboard'))
    
    if not is_valid_ticker(ticker):
        if not ticker.endswith('.NS'):
            ticker_ns = f"{ticker}.NS"
            if is_valid_ticker(ticker_ns):
                ticker = ticker_ns
            else:
                flash('Invalid stock symbol. Please enter a valid ticker like AAPL or RELIANCE.NS', 'error')
                return redirect(url_for('dashboard'))
    
    try:
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        years = min(years, app.config['MAX_TRAINING_YEARS'])
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt > end_dt:
            raise ValueError("End date must be after start date")
        
        cached = get_cached_prediction(ticker, start_date, end_date)
        if cached:
            app.logger.info(f"Using cached prediction for {ticker}")
            company_info = cached['company_info']
            currency_symbol = get_currency_symbol(company_info.get('currency', 'USD'))
            predictions = cached['predictions']
            
            historical_data = fetch_yfinance_data(ticker, years)
            
            fig = create_price_chart(predictions, ticker, currency_symbol, historical_data)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template('app.html',
                                predictions=predictions,
                                graphJSON=graphJSON,
                                company=company_info,
                                ticker=ticker,
                                currency_symbol=currency_symbol,
                                from_cache=True,
                                current_date=start_date,
                                current_end_date=end_date,
                                selected_years=years)
        
        app.logger.info(f"Making new prediction for {ticker}")
        company_info = get_company_info(ticker)
        currency_symbol = get_currency_symbol(company_info.get('currency', 'USD'))
        
        data = fetch_yfinance_data(ticker, years)
        if data.empty:
            raise ValueError("No data available for this period")
        
        X, y, scaler = prepare_training_data(data)
        if len(X) < 1:
            raise ValueError("Insufficient historical data for prediction")
        
        model = create_model(X.shape[1:])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        predictions = []
        window = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-60:].values
        window = scaler.transform(window)
        
        pred_dates = pd.date_range(start_date, end_date, freq='B')
        
        for date in pred_dates:
            pred = model.predict(np.array([window]), verbose=0)[0]
            pred_reshaped = pred.reshape(7, 5)
            prices = scaler.inverse_transform([pred_reshaped[0]])[0]
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(prices[0]),
                'high': float(prices[1]),
                'low': float(prices[2]),
                'close': float(prices[3]),
                'volume': int(prices[4]),
                'currency': currency_symbol
            })
            
            window = np.vstack([window[1:], pred_reshaped[0]])
        
        historical_data = fetch_yfinance_data(ticker, years)
        
        fig = create_price_chart(predictions, ticker, currency_symbol, historical_data)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        cache_prediction(ticker, start_date, end_date, predictions, company_info)
        
        return render_template('app.html',
                             predictions=predictions,
                             graphJSON=graphJSON,
                             company=company_info,
                             ticker=ticker,
                             currency_symbol=currency_symbol,
                             current_date=start_date,
                             current_end_date=end_date,
                             selected_years=years)
    
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('dashboard'))
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        flash("An error occurred during prediction. Please try again with a different symbol.", 'error')
        return redirect(url_for('dashboard'))

@app.route('/get_graph', methods=['POST'])
@login_required
def get_graph():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        graph_type = data.get('graphType')
        predictions = data.get('predictions', [])
        currency_symbol = data.get('currencySymbol', '$')
        
        if not ticker or not graph_type or not predictions:
            return jsonify({'error': 'Missing required data'}), 400
            
        if isinstance(predictions, str):
            predictions = json.loads(predictions)
            
        try:
            historical_data = fetch_yfinance_data(ticker, 1)
        except Exception as e:
            app.logger.warning(f"Could not fetch historical data: {str(e)}")
            historical_data = None
            
        if graph_type == 'price':
            fig = create_price_chart(predictions, ticker, currency_symbol, historical_data)
        elif graph_type == 'movingAvg':
            fig = create_moving_averages_chart(predictions, ticker, currency_symbol, historical_data)
        elif graph_type == 'warm':
            fig = create_warm_chart(predictions, ticker, currency_symbol)
        elif graph_type == 'manhattan':
            fig = create_manhattan_chart(predictions, ticker, currency_symbol)
        else:
            return jsonify({'error': 'Invalid graph type'}), 400
            
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(graphJSON)
        
    except Exception as e:
        app.logger.error(f"Error generating graph: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        
        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        email = request.form.get('email', '').strip().lower()
        
        if not all([username, password, email]):
            flash('Please fill in all fields', 'error')
            return render_template('register.html')
        
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hash_password(password), email)
            )
            db.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                flash('Username already exists', 'error')
            else:
                flash('Email already exists', 'error')
        except Exception as e:
            app.logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)