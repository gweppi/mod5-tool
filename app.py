import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="Oil & Gas Options Pricing Dashboard")

st.title(
    "Oil/Gas Option Pricing, Black-Scholes, Monte Carlo & Volatility Surface"
)

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
  z = np.random.standard_normal(num_sims)
  ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
  payoffs = np.maximum(ST - K, 0)
  return np.exp(-r * T) * np.mean(payoffs)


def implied_volatility_call(market_price, S, K, T, r):
  """Calculates Black-Scholes implied volatility using Brent's method."""
  if market_price <= max(0.0, S - K * np.exp(-r * T)):
    return np.nan
  def objective_function(sigma):
    return black_scholes_call(S, K, T, r, sigma) - market_price
  try:
    return brentq(objective_function, 0.001, 5.0)
  except ValueError:
    return np.nan


# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")

ticker_map = {
    "WTI Crude Oil (CL=F)": "CL=F",
    "Brent Crude (BZ=F)": "BZ=F",
    "Henry Hub Natural Gas (NG=F)": "NG=F",
}
selected_name = st.sidebar.selectbox("Select Asset", list(ticker_map.keys()))
ticker = ticker_map[selected_name]

spot_price = st.sidebar.number_input("Spot Price ($)", value=75.0, step=1.0)
strike_price = st.sidebar.number_input(
    "Strike Price (K) for Pricing", value=75.0, step=1.0
)
time_to_maturity = st.sidebar.slider(
    "Time to Maturity (Years)", 0.05, 2.0, 0.5, 0.05
)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.04, 0.01)
volatility = st.sidebar.slider(
    "Base Implied Volatility (sigma)", 0.05, 1.0, 0.25, 0.01
)

# --- Fetch Historical Data (5 Years) ---
@st.cache_data
def fetch_market_data(ticker_symbol):
  tk = yf.Ticker(ticker_symbol)
  return tk.history(period="5y")

data = fetch_market_data(ticker)

# --- Pricing Output Columns ---
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

# --- Implied Volatility Solver Module ---
st.subheader("Implied Volatility Solver")
col_iv1, col_iv2 = st.columns(2)
with col_iv1:
  market_option_price = st.number_input(
      "Observed Market Option Price ($)",
      value=round(bs_price, 2),
      step=0.5,
  )

calculated_iv = implied_volatility_call(
    market_option_price,
    spot_price,
    strike_price,
    time_to_maturity,
    risk_free_rate,
)

with col_iv2:
  if not np.isnan(calculated_iv):
    st.metric(
        label="Calculated Implied Volatility (IV)",
        value=f"{calculated_iv * 100:.2f}%",
    )
  else:
    st.error("Invalid market price (violates arbitrage bounds).")

# --- Historical Data Chart ---
fig_hist = px.line(
    data,
    y="Close",
    title=f"5-Year Historical Price Trend for {selected_name}",
    labels={"index": "Date", "Close": "Price ($)"},
)
min_date, max_date = data.index.min(), data.index.max()
min_price, max_price = data["Close"].min() * 0.95, data["Close"].max() * 1.05
fig_hist.update_xaxes(minallowed=min_date, maxallowed=max_date)
fig_hist.update_yaxes(minallowed=min_price, maxallowed=max_price)
st.plotly_chart(
    fig_hist, use_container_width=True, config={"displayModeBar": False}
)

# --- UPDATED: 3D Implied Volatility Surface (Smile/Skew Model) ---
st.subheader("3D Implied Volatility Surface Visualization")
st.markdown(
    "Mapping Implied Volatility across Strikes and Maturities (incorporating a"
    " volatility skew)."
)

strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 20)
expiries = np.linspace(0.05, 1.5, 20)
X, Y = np.meshgrid(strikes, expiries)

# Simulating a realistic volatility skew/smile where OTM/ITM puts/calls have higher IV
Z_iv = volatility + 0.1 * ((X - spot_price) / spot_price) ** 2 + 0.05 * (1.5 - Y)

fig_iv = go.Figure(
    data=[
        go.Surface(
            z=Z_iv,
            x=X,
            y=Y,
            colorscale="Viridis",
            hovertemplate=(
                "Strike Price: $%{x:.2f}<br>Time to Maturity: %{y:.2f}"
                " yrs<br>Implied Volatility: %{z:.2%}<extra></extra>"
            ),
        )
    ],
    layout=go.Layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Time to Maturity (Years)",
            zaxis_title="Implied Volatility (sigma)",
        ),
        autosize=True,
        height=600,
    ),
)
st.plotly_chart(
    fig_iv,
    use_container_width=True,
    config={"scrollZoom": False, "modeBarButtonsToRemove": ["zoom3d", "pan3d"]},
)

# --- UPDATED: Multi-Factor Financial Stress Test Module ---
st.subheader("Multi-Factor Financial Stress Test Module")
st.markdown(
    "Simulate market shocks across spot price, volatility, and interest rates"
    " simultaneously:"
)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
  stress_spot_drop = st.slider("Spot Price Shock (%)", -50, 50, -10, 1)
with col_s2:
  stress_vol_change = st.slider("Volatility Shock (%)", -50, 100, 20, 5)
with col_s3:
  stress_rate_change = st.slider(
      "Interest Rate Shock (bps)", -200, 200, 50, 25
  )

stressed_spot = spot_price * (1 + stress_spot_drop / 100)
stressed_vol = max(0.01, volatility * (1 + stress_vol_change / 100))
stressed_rate = max(0.0, risk_free_rate + (stress_rate_change / 10000))

stressed_bs_price = black_scholes_call(
    stressed_spot, strike_price, time_to_maturity, stressed_rate, stressed_vol
)

col3, col4 = st.columns(2)
with col3:
  st.metric(
      label="Stressed Parameters",
      value=(
          f"Spot: ${stressed_spot:.2f} | Vol: {stressed_vol*100:.1f}% | Rate:"
          f" {stressed_rate*100:.2f}%"
      ),
  )
with col4:
  st.metric(
      label="Stressed Option Value (Black-Scholes)",
      value=f"${stressed_bs_price:.2f}",
      delta=f"${stressed_bs_price - bs_price:.2f} vs Base",
  )