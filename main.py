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
# Page Config & CSS
# =========================
st.set_page_config(
    page_title="Market Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

LOGO_PATH = "/mnt/data/563923ee-2031-449c-ac30-cab848e4be18.png"  # Update path

# =========================
# Helper Functions
# =========================
def clean_ticker_input(user_ticker: str) -> str:
    """Auto-correct common ticker mistakes (S&P variants to ^GSPC)"""
    if not isinstance(user_ticker, str) or user_ticker.strip() == "":
        return user_ticker
    s = user_ticker.strip().lower().replace(" ", "").replace("-", "")
    sp500_aliases = {"s&p500","sp500","sandp500","snp500","snp","sp500"}
    if s in sp500_aliases or "sandp" in s or ("sp" in s and "500" in s):
        return "^GSPC"
    return user_ticker.strip()

@st.cache_data(ttl=60*10)
def load_data(ticker: str, days: int) -> pd.DataFrame:
    """Download data with yfinance and return a DataFrame."""
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
    """Return a Python scalar from a pandas Series or numpy type."""
    if isinstance(series_or_val, pd.Series):
        return float(series_or_val.iloc[-1])
    try:
        return float(series_or_val)
    except Exception:
        return None

def add_features(df):
    """Add target and moving average features."""
    df = df.copy()
    df['Tomorrow'] = df['Close'].shift(-1)
    tomorrow_aligned, close_aligned = df['Tomorrow'].align(df['Close'])
    df['Target'] = (tomorrow_aligned > close_aligned).astype(int)
    df.dropna(subset=['Tomorrow'], inplace=True)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df.fillna(0, inplace=True)
    return df

def train_linreg_predict(df):
    """Train LinearRegression to predict next-day Close price."""
    predictors = ['Close', 'MA5', 'MA10', 'Volume']
    df = df.copy()
    if any(col not in df.columns for col in predictors + ['Tomorrow']):
        return None
    X = df[predictors][:-1]  # Drop last row
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
    last_row = df[predictors].iloc[-1].values.reshape(1, -1)
    next_price = model.predict(last_row)[0]
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
# Sidebar
# =========================
with st.sidebar:
    st.header("Settings")
    
    # Stock selection
    ticker_input = st.text_input(
        "Enter Stock Ticker",
        value="^GSPC",
        help="e.g., ^GSPC for S&P 500, AAPL for Apple",
        key="sidebar_ticker"
    )
    ticker = clean_ticker_input(ticker_input)
    
    # Date range
    days_back = st.number_input(
        "Days Back",
        min_value=30,
        max_value=3650,
        value=365,
        key="sidebar_days_back"
    )
    
    st.markdown("---")
    st.write("Tip: Try tickers like `AAPL`, `TSLA`, `^GSPC`, or `RELIANCE.NS` (NSE with .NS).")
    
    # Refresh data
    if st.button("🔄 Refresh Data", key="refresh_button"):
        st.session_state.refresh = True
    
    # Logo
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)

# =========================
# Main Content
# =========================
st.title("📈 Market Prediction Dashboard")
st.caption("Real-time market analysis, visualization, and simple ML prediction")

# Load and process data
data = load_data(ticker, days_back)
if data is None or data.empty:
    st.error("No data available. Please check the ticker symbol or network connection.")
    st.stop()

df = add_features(data)

# Metrics
current_price = safe_scalar(df['Close'].iloc[-1])
previous_price = safe_scalar(df['Close'].iloc[0])
price_change = current_price - previous_price if previous_price not in (None, 0) else None
pct_change = (price_change / previous_price * 100) if price_change is not None else None

ma5 = safe_scalar(df['MA5'].iloc[-1])
ma10 = safe_scalar(df['MA10'].iloc[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:,.2f}", f"${price_change:,.2f}" if price_change else "")
col2.metric("% Change", f"{pct_change:.2f}%" if pct_change else "N/A")
col3.metric("MA 5", f"${ma5:,.2f}" if ma5 else "N/A")
col4.metric("MA 10", f"${ma10:,.2f}" if ma10 else "N/A")
st.divider()

# Charts: Price Trend & Volume
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Price Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], mode='lines', name='MA5', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], mode='lines', name='MA10', line=dict(dash='dash')))
    fig.update_layout(hovermode='x unified', height=450)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Volume")
    colors = ['green' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red' for i in range(1, len(df))]
    colors.insert(0, 'blue')
    fig_vol = go.Figure(go.Bar(x=df.index, y=df['Volume'], marker_color=colors))
    fig_vol.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# Statistics
st.subheader("📊 Statistics")
s1, s2, s3, s4 = st.columns(4)
s1.metric("High (period)", f"${df['High'].max():,.2f}")
s2.metric("Low (period)", f"${df['Low'].min():,.2f}")
s3.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
volatility = df['Close'].pct_change().std() * np.sqrt(252)
s4.metric("Volatility (annual)", f"{volatility*100:.2f}%")

st.divider()

# Recent Data Table
st.subheader("Recent Data")
recent = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
recent['Open'] = recent['Open'].apply(lambda v: f"${v:,.2f}")
recent['High'] = recent['High'].apply(lambda v: f"${v:,.2f}")
recent['Low'] = recent['Low'].apply(lambda v: f"${v:,.2f}")
recent['Close'] = recent['Close'].apply(lambda v: f"${v:,.2f}")
recent['Volume'] = recent['Volume'].apply(lambda v: f"{v:,.0f}")
st.dataframe(recent, use_container_width=True)

st.divider()

# Machine Learning Prediction
st.subheader("🔮 Simple ML Prediction (Next-Day Close)")
ml_result = train_linreg_predict(df)
if ml_result is None:
    st.info("Not enough data to train ML model (need at least ~20 rows after feature creation).")
else:
    st.metric("Predicted Next-Day Close", f"${ml_result['next_price']:,.2f}")
    st.write(f"Model RMSE: **{ml_result['rmse']:.4f}**  —  R²: **{ml_result['r2']:.4f}**")

    df_compare = pd.DataFrame({
        "Actual": ml_result['y_test'].values,
        "Predicted": ml_result['y_pred']
    }, index=ml_result['X_test_index'])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Actual'], name='Actual', mode='lines+markers'))
    fig2.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Predicted'], name='Predicted', mode='lines+markers'))
    fig2.update_layout(title="Actual vs Predicted (test set)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("📌 *Dashboard created with Streamlit | Data from Yahoo Finance*")
st.caption("Dashboard created with Streamlit • Data from Yahoo Finance • Simple Linear Regression model for demonstration only")
