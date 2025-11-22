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
st.write("This app loads **sp500.csv** and performs ML-based stock predictions.")

# -----------------------------------------------------
# Load CSV Data (IMPORTANT: Your CSV uses TAB separator)
# -----------------------------------------------------
@st.cache_data
def load_data(csv_path: str = "sp500.csv") -> pd.DataFrame:
    # Your file uses TAB, not comma
    df = pd.read_csv(csv_path, sep="\t")

    # Ensure Date column is processed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    return df

# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Close" not in df.columns:
        st.error("❌ Your CSV must have a 'Close' column.")
        st.write("Columns found:", df.columns)
        raise ValueError("CSV must contain a Close column")

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()

    df = df.dropna()
    return df

# -----------------------------------------------------
# ML Training
# -----------------------------------------------------
def train_model(df):
    features = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]

    X = df[features]
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
# Load, Process & Display Data
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
# Show Metrics
# -----------------------------------------------------
st.subheader("📉 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")

# -----------------------------------------------------
# Plot Actual vs Predicted
# -----------------------------------------------------
st.subheader("📊 Actual vs Predicted Close Price")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_index,
    y=df.loc[pred_index, "Tomorrow"],
    mode='lines',
    name='Actual'
))
fig.add_trace(go.Scatter(
    x=pred_index,
    y=predictions,
    mode='lines',
    name='Predicted'
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Predict Next Day
# -----------------------------------------------------
st.subheader("🔮 Predict Next Day Price")

last_row = df.iloc[-1:]
next_pred = model.predict(last_row[["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]])[0]

st.success(f"📌 Predicted Next Day Close Price: **{next_pred:.2f}**")
