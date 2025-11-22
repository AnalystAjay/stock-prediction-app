import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# -----------------------------------------------------
# Streamlit Config
# -----------------------------------------------------
st.set_page_config(page_title="Stock Prediction App (CSV)", layout="wide")
st.title("📈 Stock Prediction App (CSV Based)")
st.write("This app loads **sp500.csv** and performs ML-based stock prediction.")

# -----------------------------------------------------
# Load CSV Data
# -----------------------------------------------------
@st.cache_data
def load_data(csv_path: str = "sp500.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Fix date column if exists
    date_cols = [c for c in ["Date", "date"] if c in df.columns]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.sort_values(date_cols[0])
        df = df.set_index(date_cols[0])
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Fix OHLC column names
    expected = ["Open", "High", "Low", "Close", "Volume"]
    available = {c.lower(): c for c in df.columns}

    for e in expected:
        if e not in df.columns and e.lower() in available:
            df = df.rename(columns={available[e.lower()]: e})

    return df


# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Close" not in df.columns:
        raise ValueError("CSV must contain a Close column")

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    # Basic ML features
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()

    df = df.dropna()

    return df


# -----------------------------------------------------
# ML Training Function
# -----------------------------------------------------
def train_model(df):
    X = df[["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]]
    y = df["Tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, X_test.index, predictions, mse, r2


# -----------------------------------------------------
# Load and Process Data
# -----------------------------------------------------
df = load_data()
df = add_features(df)

st.subheader("📄 Raw Data Preview")
st.dataframe(df.head())

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------
model, pred_index, predictions, mse, r2 = train_model(df)

# -----------------------------------------------------
# Display Results
# -----------------------------------------------------
st.subheader("📉 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")

# -----------------------------------------------------
# Plot Predictions
# -----------------------------------------------------
st.subheader("📊 Actual vs Predicted Close Price")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_index,
    y=df.loc[pred_index, "Tomorrow"],
    mode='lines',
    name="Actual"
))
fig.add_trace(go.Scatter(
    x=pred_index,
    y=predictions,
    mode='lines',
    name="Predicted"
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Predict Next Day Price
# -----------------------------------------------------
st.subheader("🧮 Predict Next Day Price")

last_row = df.iloc[-1:].copy()
next_pred = model.predict(last_row[["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]])[0]

st.success(f"📌 **Predicted Next Day Close Price:** `{next_pred:.2f}`")
