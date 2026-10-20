import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(layout="wide", page_title="Oil & Gas Options Pricing Dashboard")

st.title(
    "Oil/Gas Option Pricing, Black-Scholes, Monte Carlo & Implied Volatility"
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
  """Calculates the Black-Scholes implied volatility given a market option price

  using Brent's root-finding method.
  """
  # Check basic arbitrage bounds
  if market_price <= max(0.0, S - K * np.exp(-r * T)):
    return np.nan

  def objective_function(sigma):
    return black_scholes_call(S, K, T, r, sigma) - market_price

  try:
    # Search for volatility between 0.1% and 500%
    iv = brentq(objective_function, 0.001, 5.0)
    return iv
  except ValueError:
    return np.nan


# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
# Map human-readable names to yfinance ticker symbols
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
    "Input Volatility (sigma) for Pricing", 0.05, 1.0, 0.25, 0.01
)

# --- Fetch Historical Data ---
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

# --- NEW: Implied Volatility Solver Module ---
st.subheader("Implied Volatility Solver")
st.markdown(
    "Input an observed market option price below to calculate its Black-Scholes"
    " Implied Volatility:"
)

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
    st.error(
        "Invalid market price (violates arbitrage bounds or cannot converge)."
    )

# --- Historical Data Chart ---
st.subheader(f"Historical Price Trend for {selected_name}")
fig_hist = px.line(
    data,
    y="Close",
    title=f"5-Year Historical Price Trend for {selected_name}",
    labels={"index": "Date", "Close": "Price ($)"},
)

min_date = data.index.min()
max_date = data.index.max()
min_price = data["Close"].min() * 0.95
max_price = data["Close"].max() * 1.05

fig_hist.update_xaxes(minallowed=min_date, maxallowed=max_date)
fig_hist.update_yaxes(minallowed=min_price, maxallowed=max_price)

st.plotly_chart(
    fig_hist, use_container_width=True, config={"displayModeBar": False}
)

# --- 3D Option Pricing Surface ---
st.subheader("3D Black-Scholes Option Pricing Surface")

strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 20)
expiries = np.linspace(0.05, 1.5, 20)
X, Y = np.meshgrid(strikes, expiries)

Z = np.array(
    [
        black_scholes_call(spot_price, k, t, risk_free_rate, volatility)
        for k, t in zip(X.ravel(), Y.ravel())
    ]
).reshape(X.shape)

fig = go.Figure(
    data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', hovertemplate=(
                "Strike Price: $%{x:.2f}<br>Time to Maturity: %{y:.2f}"
                " yrs<br>Option Price: $%{z:.2f}<extra></extra>"
            ),)],
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
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["zoom3d", "pan3d"]})

# --- Financial Stress Testing Module ---
st.subheader("Financial Stress Test Module")
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
