import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Stock Prediction App (CSV)", layout="wide")


# -----------------------------
# Data loading (CSV only)
# -----------------------------
@st.cache_data
def load_data(csv_path: str = "sp500.csv") -> pd.DataFrame:
df = pd.read_csv(csv_path)


# Accept common date column names
date_cols = [c for c in ["Date", "date"] if c in df.columns]
if date_cols:
df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
df = df.sort_values(date_cols[0])
df = df.set_index(date_cols[0])
else:
# if no date column, create a simple range index
df.index = pd.RangeIndex(start=0, stop=len(df), step=1)


# Ensure expected columns exist or rename if needed
expected = ["Open", "High", "Low", "Close", "Volume"]
# if your CSV has lowercase or different names, adapt here
available = {c.lower(): c for c in df.columns}
for e in expected:
if e not in df.columns and e.lower() in available:
df = df.rename(columns={available[e.lower()]: e})


return df




# -----------------------------
# Feature engineering
# -----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
if "Close" not in df.columns:
raise ValueError("CSV must contain a Close column")


df["Tomorrow"] = df["Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
st.write(results["features"])
