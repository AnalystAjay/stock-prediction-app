import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="Market Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
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

# Title
st.title("📈 Market Prediction Dashboard")
st.markdown("Real-time market analysis and visualization")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Stock selection
    ticker = st.text_input("Enter Stock Ticker", value="^GSPC", help="e.g., ^GSPC for S&P 500, AAPL for Apple")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.number_input("Days Back", min_value=30, max_value=365, value=90)
    
    # Refresh data
    if st.button("🔄 Refresh Data"):
        st.session_state.refresh = True

# Load data
@st.cache_data
def load_data(ticker, days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main content
try:
    # Load stock data
    data = load_data(ticker, days_back)
    
    if data is not None and len(data) > 0:
        # Calculate metrics
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[0]
        price_change = current_price - previous_price
        pct_change = (price_change / previous_price) * 100
        
        ma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"${price_change:.2f}")
        
        with col2:
            st.metric("% Change", f"{pct_change:.2f}%", delta_color="inverse" if pct_change < 0 else "normal")
        
        with col3:
            st.metric("MA 20", f"${ma_20:.2f}")
        
        with col4:
            st.metric("MA 50", f"${ma_50:.2f}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price Trend")
            
            # Create price chart with moving averages
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'].rolling(window=20).mean(),
                mode='lines',
                name='MA 20',
                line=dict(color='#ff7f0e', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'].rolling(window=50).mean(),
                mode='lines',
                name='MA 50',
                line=dict(color='#d62728', dash='dash')
            ))
            
            fig.update_layout(
                hovermode='x unified',
                height=400,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Volume")
            
            # Volume chart
            fig_vol = go.Figure()
            
            colors = ['green' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'red' 
                     for i in range(1, len(data))]
            colors.insert(0, 'blue')
            
            fig_vol.add_trace(go.Bar(
                x=data.index, y=data['Volume'],
                marker_color=colors,
                name='Volume',
                hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
            ))
            
            fig_vol.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("High (Period)", f"${data['High'].max():.2f}")
        
        with stats_col2:
            st.metric("Low (Period)", f"${data['Low'].min():.2f}")
        
        with stats_col3:
            st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
        
        with stats_col4:
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            st.metric("Volatility (Annual)", f"{volatility*100:.2f}%")
        
        # Data table
        st.subheader("Recent Data")
        st.dataframe(
            data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].apply(
                lambda x: [f"${v:.2f}" if x.name in ['Open', 'High', 'Low', 'Close'] else f"{v:,.0f}" for v in x]
            ),
            use_container_width=True
        )
        
    else:
        st.error("No data available. Please check the ticker symbol.")

except Exception as e:
    st.error(f"An error occurred: {e}")

# Footer
st.divider()
st.markdown("---")
st.markdown("📌 *Dashboard created with Streamlit | Data from Yahoo Finance*")
