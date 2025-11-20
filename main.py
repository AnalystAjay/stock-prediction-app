import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# =========================
# Config & helper functions
# =========================
st.set_page_config(page_title="Market Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

LOGO_PATH = "/mnt/data/563923ee-2031-449c-ac30-cab848e4be18.png"  # your uploaded image path

def clean_ticker_input(user_ticker: str) -> str:
    """Auto-correct common ticker mistakes (S&P variants to ^GSPC)"""
    if not isinstance(user_ticker, str) or user_ticker.strip() == "":
        return user_ticker
    s = user_ticker.strip().lower().replace(" ", "").replace("-", "")
    sp500_aliases = {"s&p500","sp500","sandp500","snp500","snp","sp500"}
    if s in sp500_aliases or "sandp" in s or "sp" in s and "500" in s:
        return "^GSPC"
    return user_ticker.strip()

@st.cache_data(ttl=60*10)
def load_data(ticker: str, days: int) -> pd.DataFrame:
    """Download data with yfinance and return a DataFrame (or None if failed)."""
    if ticker is None or ticker == "":
        return None
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def safe_scalar(series_or_val):
    """Return a python scalar (float/int) from a pandas Series or numpy type for formatting."""
    if isinstance(series_or_val, pd.Series):
        return float(series_or_val.iloc[-1])
    try:
        return float(series_or_val)
    except Exception:
        return None

def add_features(df):
    df = df.copy()

    # Tomorrow price
    df['Tomorrow'] = df['Close'].shift(-1)

    # Example indicators (your code may vary)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()

    # Drop rows with NaN values caused by rolling or shift
    df = df.dropna()

    # FIX: Reset index before comparing
    df = df.reset_index(drop=True)

    # Create Target column
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

    return df

def train_linreg_predict(df):
    """Train simple LinearRegression to predict next-day Close price and return metrics and next-day prediction."""
    # Use features: Close, MA20, MA50, Volume
    predictors = ['Close', 'MA20', 'MA50', 'Volume']
    df = df.copy()
    if any(col not in df.columns for col in predictors + ['Tomorrow']):
        return None

    X = df[predictors].shift(0)[:-1]   # drop last row since Tomorrow is shift(-1)
    y = df['Tomorrow'][:-1]
    if len(X) < 10:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # predict next day using the last available row's features
    last_row = df[predictors].iloc[-1].values.reshape(1, -1)
    next_price = model.predict(last_row)[0]

    # return model info
    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "next_price": float(next_price),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test_index": X_test.index
    }

# =========================
# UI - Sidebar
# =========================
with st.sidebar:
    st.header("Settings")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)
    ticker_input = st.text_input("Enter Stock Ticker", value="^GSPC", help="e.g., ^GSPC for S&P 500, AAPL for Apple")
    ticker = clean_ticker_input(ticker_input)
    days_back = st.number_input("Days Back", min_value=30, max_value=3650, value=365)
    st.markdown("---")
    st.write("Tip: Try tickers like `AAPL`, `TSLA`, `^GSPC`, or `RELIANCE.NS` (NSE with .NS).")
    if st.button("🔄 Refresh Data"):
        st.session_state.refresh = True

# Header area
st.title("📈 Market Prediction Dashboard")
st.caption("Real-time market analysis, visualization, and simple ML prediction")

# =========================
# Main
# =========================
data = load_data(ticker, days_back)

if data is None or data.empty:
    st.error("No data available. Please check the ticker symbol or network connection.")
    st.stop()

# Add features for ML
df = add_features(data)

# Basic metrics - convert Series -> scalar before formatting
try:
    current_price = safe_scalar(df['Close'].iloc[-1])
    previous_price = safe_scalar(df['Close'].iloc[0])
    price_change = None
    pct_change = None
    if current_price is not None and previous_price not in (None, 0):
        price_change = current_price - previous_price
        pct_change = (price_change / previous_price) * 100
except Exception:
    current_price = None
    price_change = None
    pct_change = None

ma_20 = safe_scalar(df['MA20'].iloc[-1]) if 'MA20' in df.columns else None
ma_50 = safe_scalar(df['MA50'].iloc[-1]) if 'MA50' in df.columns else None

# Metrics cards
col1, col2, col3, col4 = st.columns(4)
if current_price is not None:
    col1.metric("Current Price", f"${current_price:,.2f}", f"${price_change:,.2f}" if price_change is not None else "")
else:
    col1.metric("Current Price", "N/A")

if pct_change is not None:
    col2.metric("% Change (period)", f"{pct_change:.2f}%")
else:
    col2.metric("% Change (period)", "N/A")

col3.metric("MA 20", f"${ma_20:,.2f}" if ma_20 is not None else "N/A")
col4.metric("MA 50", f"${ma_50:,.2f}" if ma_50 is not None else "N/A")

st.divider()

# Charts
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Price Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(), mode='lines', name='MA20', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50', line=dict(dash='dash')))
    fig.update_layout(hovermode='x unified', height=450)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Volume")
    colors = ['green' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'red' for i in range(1, len(data))]
    colors.insert(0, 'blue')
    fig_vol = go.Figure(go.Bar(x=data.index, y=data['Volume'], marker_color=colors))
    fig_vol.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# Statistics
st.subheader("📊 Statistics")
s1, s2, s3, s4 = st.columns(4)
s1.metric("High (period)", f"${data['High'].max():,.2f}")
s2.metric("Low (period)", f"${data['Low'].min():,.2f}")
s3.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
volatility = data['Close'].pct_change().std() * np.sqrt(252)
s4.metric("Volatility (annual)", f"{volatility*100:.2f}%")

st.divider()

# Recent Data table (formatted safely)
st.subheader("Recent Data")
recent = data.tail(10).copy()
recent_display = recent[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
recent_display['Open'] = recent_display['Open'].apply(lambda v: f"${v:,.2f}")
recent_display['High'] = recent_display['High'].apply(lambda v: f"${v:,.2f}")
recent_display['Low'] = recent_display['Low'].apply(lambda v: f"${v:,.2f}")
recent_display['Close'] = recent_display['Close'].apply(lambda v: f"${v:,.2f}")
recent_display['Volume'] = recent_display['Volume'].apply(lambda v: f"{v:,.0f}")
st.dataframe(recent_display, use_container_width=True)

st.divider()

# =========================
# Machine Learning Prediction (Linear Regression)
# =========================
st.subheader("🔮 Simple ML Prediction (Next-Day Close)")

ml_result = train_linreg_predict(df)

if ml_result is None:
    st.info("Not enough data to train ML model (need at least ~20 rows after feature creation).")
else:
    next_price = ml_result['next_price']
    rmse = ml_result['rmse']
    r2 = ml_result['r2']

    # show predicted price as metric
    st.metric("Predicted Next-Day Close", f"${next_price:,.2f}")

    # show model performance
    st.write(f"Model RMSE: **{rmse:,.4f}**  —  R²: **{r2:.4f}**")

    # plot predicted vs actual for test slice
    y_test = ml_result['y_test']
    y_pred = ml_result['y_pred']
    x_idx = ml_result['X_test_index']

    df_compare = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=x_idx)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Actual'], name='Actual', mode='lines+markers'))
    fig2.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Predicted'], name='Predicted', mode='lines+markers'))
    fig2.update_layout(title="Actual vs Predicted (test set)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Dashboard created with Streamlit • Data from Yahoo Finance • Simple Linear Regression model for demonstration only")
