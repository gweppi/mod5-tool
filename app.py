import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="Oil & Gas Options Pricing Dashboard")

st.title("🛢️ Oil/Gas Option Pricing, Black-Scholes & Monte Carlo Dashboard")

# --- 1. Math Functions ---


def black_scholes_call(S, K, T, r, sigma):
  """Calculates the Black-Scholes European Call Option Price."""
  if T <= 0 or sigma <= 0:
    return max(0.0, S - K)
  d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def monte_carlo_call(S, K, T, r, sigma, num_sims=50000):
  """Calculates European Call Option Price via Monte Carlo simulation."""
  if T <= 0:
    return max(0.0, S - K)
  # Generate standard normal random variables
  z = np.random.standard_normal(num_sims)
  # Simulate terminal stock prices under GBM
  ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
  # Calculate payoffs and discount back to present value
  payoffs = np.maximum(ST - K, 0)
  price = np.exp(-r * T) * np.mean(payoffs)
  return price


# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
ticker = st.sidebar.selectbox("Select Asset", ["CL=F", "BZ=F", "NG=F"])  # Crude, Brent, Natural Gas
spot_price = st.sidebar.number_input("Spot Price ($)", value=75.0, step=1.0)
strike_price = st.sidebar.number_input(
    "Strike Price (K) for Pricing", value=75.0, step=1.0
)
time_to_maturity = st.sidebar.slider(
    "Time to Maturity (Years)", 0.05, 2.0, 0.5, 0.05
)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.04, 0.01)
volatility = st.sidebar.slider(
    "Implied Volatility (sigma)", 0.05, 1.0, 0.25, 0.01
)

# --- Fetch Historical Data ---
@st.cache_data
def fetch_market_data(ticker_symbol):
  tk = yf.Ticker(ticker_symbol)
  return tk.history(period="6mo")


data = fetch_market_data(ticker)

# --- Dashboard Layout: Columns for Pricing Output ---
col1, col2 = st.columns(2)

bs_price = black_scholes_call(
    spot_price, strike_price, time_to_maturity, risk_free_rate, volatility
)
mc_price = monte_carlo_call(
    spot_price, strike_price, time_to_maturity, risk_free_rate, volatility
)

with col1:
  st.metric(
      label="Black-Scholes Call Price (Theoretical)",
      value=f"${bs_price:.2f}",
  )

with col2:
  st.metric(
      label="Monte Carlo Call Price (Simulation)",
      value=f"${mc_price:.2f}",
      delta=f"{((mc_price - bs_price) / bs_price) * 100:.2f}% vs BS",
  )

# --- Historical Data Chart ---
st.subheader(f"Historical Price Trend for {ticker}")
st.line_chart(data["Close"])

# --- 3D Option Pricing Surface (Black-Scholes Model) ---
st.subheader("3D Black-Scholes Option Pricing Surface")

# Create grid for Strike Prices and Maturities
strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 20)
expiries = np.linspace(0.05, 1.5, 20)
X, Y = np.meshgrid(strikes, expiries)

# Compute BS price for every point on the grid
Z = np.array(
    [
        black_scholes_call(spot_price, k, t, risk_free_rate, volatility)
        for k, t in zip(X.ravel(), Y.ravel())
    ]
).reshape(X.shape)

fig = go.Figure(
    data=[go.Surface(z=Z, x=X, y=Y)],
    layout=go.Layout(
        title="Call Option Price Surface",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Time to Maturity (Years)",
            zaxis_title="Option Price ($)",
        ),
        autosize=True,
        height=600,
    ),
)

st.plotly_chart(fig, use_container_width=True)

# --- Financial Stress Testing Module ---
st.subheader("⚠️ Financial Stress Test Module")
stress_drop = st.slider("Simulated Underlying Price Drop (%)", 0, 50, 10)
stressed_spot = spot_price * (1 - stress_drop / 100)
stressed_bs_price = black_scholes_call(
    stressed_spot, strike_price, time_to_maturity, risk_free_rate, volatility
)

col3, col4 = st.columns(2)
with col3:
  st.metric(
      label="Stressed Spot Price",
      value=f"${stressed_spot:.2f}",
      delta=f"-{stress_drop}%",
  )
with col4:
  st.metric(
      label="Stressed Option Value",
      value=f"${stressed_bs_price:.2f}",
      delta=f"${stressed_bs_price - bs_price:.2f}",
  )
