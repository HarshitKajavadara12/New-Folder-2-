"""
GIGA SYSTEM - Basic Usage Example
Demonstrates core functionality
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# 1. OPTIONS PRICING
# =============================================================================

print("=" * 60)
print("GIGA SYSTEM - Basic Example")
print("=" * 60)

# Import core modules
from giga_system.core import (
    black_scholes_price,
    BlackScholesModel,
    OptionGreeks
)

# Option parameters
S = 100      # Spot price
K = 100      # Strike price
T = 0.25     # Time to expiry (3 months)
r = 0.05     # Risk-free rate (5%)
sigma = 0.20 # Volatility (20%)

# Price a call option
call_price = black_scholes_price(S, K, T, r, sigma, 'call')
put_price = black_scholes_price(S, K, T, r, sigma, 'put')

print("\n  Option Pricing")
print("-" * 40)
print(f"Spot Price:     ${S:.2f}")
print(f"Strike Price:   ${K:.2f}")
print(f"Time to Expiry: {T*12:.1f} months")
print(f"Risk-Free Rate: {r*100:.1f}%")
print(f"Volatility:     {sigma*100:.1f}%")
print("-" * 40)
print(f"Call Price:     ${call_price:.4f}")
print(f"Put Price:      ${put_price:.4f}")

# Verify put-call parity
parity_lhs = call_price + K * np.exp(-r * T)
parity_rhs = put_price + S
print(f"\nPut-Call Parity: {parity_lhs:.4f} ≈ {parity_rhs:.4f}  ")


# =============================================================================
# 2. GREEKS CALCULATION
# =============================================================================

print("\n\n  Greeks Analysis")
print("-" * 40)

greeks = OptionGreeks()

# Calculate all Greeks for the call
delta = greeks.delta(S, K, T, r, sigma, 'call')
gamma = greeks.gamma(S, K, T, r, sigma)
theta = greeks.theta(S, K, T, r, sigma, 'call')
vega = greeks.vega(S, K, T, r, sigma)
rho = greeks.rho(S, K, T, r, sigma, 'call')

print("Call Option Greeks:")
print(f"  Delta (Δ):  {delta:.4f}   (Price changes ${delta:.2f} per $1 spot move)")
print(f"  Gamma (Γ):  {gamma:.4f}   (Delta changes {gamma:.4f} per $1 spot move)")
print(f"  Theta (Θ): {theta:.4f}   (Loses ${-theta*100:.2f} per day)")
print(f"  Vega  (V):  {vega:.4f}   (Gains ${vega*100:.2f} per 1% vol increase)")
print(f"  Rho   (ρ):  {rho:.4f}   (Gains ${rho*100:.2f} per 1% rate increase)")


# =============================================================================
# 3. VECTORIZED CALCULATIONS
# =============================================================================

print("\n\n  Vectorized Greeks (1000 options)")
print("-" * 40)

# Price 1000 options at different spots
spots = np.linspace(80, 120, 1000)

import time
start = time.perf_counter()

# Vectorized calculation
bs = BlackScholesModel(S=spots, K=K, T=T, r=r, sigma=sigma)
call_prices = bs.call_price()
deltas = bs.delta('call')
gammas = bs.gamma()

elapsed = (time.perf_counter() - start) * 1000

print(f"Calculated {len(spots)} options in {elapsed:.2f} ms")
print(f"Average: {elapsed/len(spots)*1000:.2f} μs per option")


# =============================================================================
# 4. RISK METRICS
# =============================================================================

print("\n\n ️ Risk Analysis")
print("-" * 40)

from giga_system.core import RiskMetrics

# Generate sample returns
np.random.seed(42)
daily_returns = np.random.randn(252) * 0.02 + 0.0003  # 252 trading days

risk = RiskMetrics()

# Calculate risk metrics
var_95 = risk.var_historical(daily_returns, confidence=0.95, horizon=1)
var_99 = risk.var_historical(daily_returns, confidence=0.99, horizon=1)
cvar_95 = risk.cvar(daily_returns, confidence=0.95)
max_dd = risk.max_drawdown(daily_returns)

print("Portfolio Risk Metrics (1 day horizon):")
print(f"  VaR 95%:      {var_95*100:.2f}%")
print(f"  VaR 99%:      {var_99*100:.2f}%")
print(f"  CVaR 95%:     {cvar_95*100:.2f}%")
print(f"  Max Drawdown: {max_dd*100:.2f}%")

# Risk-adjusted returns
sharpe = risk.sharpe_ratio(daily_returns, rf_rate=0.02/252)
sortino = risk.sortino_ratio(daily_returns, rf_rate=0.02/252)

print(f"\nRisk-Adjusted Performance:")
print(f"  Sharpe Ratio:  {sharpe:.2f}")
print(f"  Sortino Ratio: {sortino:.2f}")


# =============================================================================
# 5. MONTE CARLO SIMULATION
# =============================================================================

print("\n\n  Monte Carlo Pricing")
print("-" * 40)

from giga_system.core import MonteCarloEngine, PathGenerator

# Generate price paths
pg = PathGenerator(S0=100, r=0.05, sigma=0.2, T=0.25, n_steps=63, n_paths=10000)
paths = pg.gbm()

print(f"Generated {paths.shape[0]} paths with {paths.shape[1]} steps each")

# Price European call via MC
mc = MonteCarloEngine(n_paths=100000)
mc_price = mc.price_european(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')

print(f"\nEuropean Call (MC): ${mc_price:.4f}")
print(f"European Call (BS): ${call_price:.4f}")
print(f"Difference:         ${abs(mc_price - call_price):.4f}")


# =============================================================================
# 6. SIMPLE STRATEGY
# =============================================================================

print("\n\n  Simple Momentum Strategy")
print("-" * 40)

from giga_system.strategies import MomentumStrategy
from giga_system.data import TechnicalIndicators

# Generate sample price data
dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
prices = pd.DataFrame({
    'open': 100 + np.cumsum(np.random.randn(252) * 0.5),
    'high': 0,
    'low': 0,
    'close': 0,
    'volume': np.random.randint(100000, 1000000, 252)
}, index=dates)
prices['close'] = prices['open'] + np.random.randn(252) * 0.3
prices['high'] = prices[['open', 'close']].max(axis=1) + abs(np.random.randn(252) * 0.2)
prices['low'] = prices[['open', 'close']].min(axis=1) - abs(np.random.randn(252) * 0.2)

# Add indicators
indicators = TechnicalIndicators()
prices['sma_20'] = indicators.sma(prices['close'], 20)
prices['rsi'] = indicators.rsi(prices['close'], 14)

print("Price Data Sample:")
print(prices.tail().to_string())

# Initialize strategy
strategy = MomentumStrategy(
    lookback=20,
    threshold=0.02,
    risk_per_trade=0.01
)

# Generate signals
signals = strategy.generate_signals(prices.dropna())

print(f"\nGenerated {len(signals)} signals")
buy_signals = sum(1 for s in signals if s.direction == 1)
sell_signals = sum(1 for s in signals if s.direction == -1)
print(f"  Buy signals:  {buy_signals}")
print(f"  Sell signals: {sell_signals}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n\n" + "=" * 60)
print("  Basic Example Complete!")
print("=" * 60)
print("""
Next steps:
1. Run the dashboard:  streamlit run visualization/app.py
2. Try advanced examples: python docs/examples/advanced_example.py
3. Read the docs: docs/API_REFERENCE.md
""")
