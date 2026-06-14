"""
GIGA SYSTEM - Advanced Usage Example
Demonstrates advanced features including R analytics and quantum computing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("GIGA SYSTEM - Advanced Example")
print("=" * 70)


# =============================================================================
# 1. PAIRS TRADING WITH COINTEGRATION
# =============================================================================

print("\n\n  PAIRS TRADING - Cointegration Analysis")
print("-" * 60)

from giga_system.strategies import PairsFinder, StatArbStrategy

# Generate cointegrated pair data
np.random.seed(42)
n = 500

# Generate a common stochastic trend
trend = np.cumsum(np.random.randn(n) * 0.5)

# Asset A follows trend closely
asset_a = 100 + trend + np.random.randn(n) * 0.5

# Asset B follows trend with mean-reverting spread
spread_mean = 5
spread = np.zeros(n)
spread[0] = spread_mean
for i in range(1, n):
    spread[i] = spread[i-1] + 0.3 * (spread_mean - spread[i-1]) + np.random.randn() * 0.3

asset_b = 100 + trend + spread + np.random.randn(n) * 0.3

prices = pd.DataFrame({
    'STOCK_A': asset_a,
    'STOCK_B': asset_b
}, index=pd.date_range('2023-01-01', periods=n, freq='D'))

# Find cointegrated pairs
finder = PairsFinder()
pairs = finder.find_pairs(prices, significance=0.05)

print(f"Found {len(pairs)} cointegrated pairs")
if pairs:
    for pair in pairs:
        print(f"  {pair['asset1']} - {pair['asset2']}")
        print(f"    Hedge Ratio: {pair['hedge_ratio']:.4f}")
        print(f"    Half-life:   {pair['halflife']:.1f} days")
        print(f"    P-value:     {pair['pvalue']:.4f}")

# Calculate spread and z-score
spread = prices['STOCK_A'] - pair['hedge_ratio'] * prices['STOCK_B']
spread_zscore = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

print(f"\nSpread Statistics:")
print(f"  Mean:     {spread.mean():.2f}")
print(f"  Std Dev:  {spread.std():.2f}")
print(f"  Current Z-Score: {spread_zscore.iloc[-1]:.2f}")

# Run strategy
strat = StatArbStrategy(
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5,
    stop_zscore=4.0
)

signals = strat.generate_signals(spread_zscore.dropna())
print(f"\nGenerated {len(signals)} trading signals")


# =============================================================================
# 2. GARCH VOLATILITY MODELING (R Analytics)
# =============================================================================

print("\n\n  GARCH VOLATILITY MODELING")
print("-" * 60)

try:
    from giga_system.bridge import RBridge
    
    # Initialize R bridge
    r = RBridge()
    
    # Calculate returns
    returns = prices['STOCK_A'].pct_change().dropna().values
    
    # Fit GARCH(1,1) model
    print("Fitting GARCH(1,1) model...")
    garch_result = r.fit_garch(returns, p=1, q=1, dist='std')
    
    print(f"\nGARCH(1,1) Parameters:")
    print(f"  μ (mean):      {garch_result['mu']:.6f}")
    print(f"  ω (omega):     {garch_result['omega']:.6f}")
    print(f"  α (alpha1):    {garch_result['alpha1']:.4f}")
    print(f"  β (beta1):     {garch_result['beta1']:.4f}")
    print(f"  ν (shape):     {garch_result['shape']:.2f} (t-dist degrees)")
    
    # Check stationarity
    persistence = garch_result['alpha1'] + garch_result['beta1']
    print(f"\n  Persistence (α+β): {persistence:.4f}")
    print(f"  Model {'is' if persistence < 1 else 'is NOT'} stationary")
    
    # Forecast volatility
    forecast = r.forecast_garch(garch_result, horizon=10)
    print(f"\n10-Day Volatility Forecast:")
    for i, vol in enumerate(forecast['sigma'][:5], 1):
        print(f"  Day {i}: {vol*100*np.sqrt(252):.2f}% (annualized)")
    
except ImportError:
    print("R analytics not available (rpy2 not installed)")
    print("Simulating GARCH output...")
    
    # Simulated GARCH parameters
    print(f"\nSimulated GARCH(1,1) Parameters:")
    print(f"  μ (mean):      0.000500")
    print(f"  ω (omega):     0.000010")
    print(f"  α (alpha1):    0.0850")
    print(f"  β (beta1):     0.9050")
    print(f"  Persistence:   0.9900")


# =============================================================================
# 3. REGIME DETECTION (Hidden Markov Model)
# =============================================================================

print("\n\n  REGIME DETECTION - Hidden Markov Model")
print("-" * 60)

try:
    # Fit HMM
    hmm_result = r.fit_hmm(returns, n_states=2)
    
    print("2-State HMM Results:")
    print(f"\nState Means (annualized returns):")
    for i, mean in enumerate(hmm_result['means']):
        regime = "Bull  " if mean > 0 else "Bear  "
        print(f"  State {i+1} ({regime}): {mean*252*100:.1f}%")
    
    print(f"\nState Volatilities (annualized):")
    for i, vol in enumerate(hmm_result['volatilities']):
        print(f"  State {i+1}: {vol*np.sqrt(252)*100:.1f}%")
    
    print(f"\nTransition Matrix:")
    print(hmm_result['transition_matrix'])
    
    print(f"\nCurrent Regime: State {hmm_result['current_state'] + 1}")
    
except:
    print("Simulated HMM Results:")
    print(f"\n  State 1 (Bull  ): +15.2% annualized")
    print(f"  State 2 (Bear  ): -8.5% annualized")
    print(f"\nTransition Probabilities:")
    print(f"  P(Bull → Bull) = 0.95")
    print(f"  P(Bear → Bear) = 0.92")


# =============================================================================
# 4. PORTFOLIO OPTIMIZATION
# =============================================================================

print("\n\n  PORTFOLIO OPTIMIZATION")
print("-" * 60)

from giga_system.core import RiskMetrics

# Generate multi-asset returns
np.random.seed(42)
n_assets = 5
n_days = 252
asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']

# Expected returns and volatilities
exp_returns = np.array([0.15, 0.12, 0.14, 0.18, 0.22])  # Annualized
volatilities = np.array([0.20, 0.18, 0.17, 0.25, 0.30])

# Correlation matrix
corr = np.array([
    [1.00, 0.65, 0.70, 0.55, 0.50],
    [0.65, 1.00, 0.68, 0.52, 0.48],
    [0.70, 0.68, 1.00, 0.58, 0.52],
    [0.55, 0.52, 0.58, 1.00, 0.62],
    [0.50, 0.48, 0.52, 0.62, 1.00]
])

# Covariance matrix
cov_matrix = np.outer(volatilities, volatilities) * corr

print("Asset Universe:")
print("-" * 50)
for i, name in enumerate(asset_names):
    print(f"  {name}: E[R]={exp_returns[i]*100:.1f}%, Vol={volatilities[i]*100:.1f}%")

# Mean-Variance Optimization
def optimize_portfolio(exp_returns, cov_matrix, risk_aversion=2.0):
    """Simple mean-variance optimization."""
    n = len(exp_returns)
    
    # Quadratic programming solution approximation
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Optimal weights (unconstrained)
    weights = inv_cov @ exp_returns / risk_aversion
    
    # Normalize and clip
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    
    return weights

# Optimize with different risk aversions
print("\nOptimal Portfolios:")
print("-" * 50)

for risk_aversion in [1.0, 2.0, 5.0]:
    weights = optimize_portfolio(exp_returns, cov_matrix, risk_aversion)
    
    port_return = np.dot(weights, exp_returns)
    port_vol = np.sqrt(np.dot(weights, cov_matrix @ weights))
    sharpe = port_return / port_vol
    
    print(f"\nRisk Aversion = {risk_aversion}")
    print(f"  Expected Return: {port_return*100:.1f}%")
    print(f"  Volatility:      {port_vol*100:.1f}%")
    print(f"  Sharpe Ratio:    {sharpe:.2f}")
    print(f"  Weights: {dict(zip(asset_names, [f'{w:.1%}' for w in weights]))}")


# =============================================================================
# 5. QUANTUM PORTFOLIO OPTIMIZATION
# =============================================================================

print("\n\n  QUANTUM PORTFOLIO OPTIMIZATION")
print("-" * 60)

try:
    from giga_system.quantum import QuantumOptimizer, QuantumPortfolio
    
    # Initialize quantum portfolio optimizer
    qp = QuantumPortfolio(
        expected_returns=exp_returns,
        covariance=cov_matrix,
        risk_aversion=2.0
    )
    
    print("Running QAOA optimization...")
    
    # Optimize (select 3 out of 5 assets)
    result = qp.optimize(method='qaoa', budget=3, depth=2, shots=1024)
    
    print(f"\nQuantum Optimization Result:")
    print(f"  Selected Assets: {[asset_names[i] for i in result['selection']]}")
    print(f"  Weights: {result['weights']}")
    print(f"  Expected Return: {result['expected_return']*100:.1f}%")
    print(f"  Expected Risk: {result['expected_risk']*100:.1f}%")
    print(f"  Cost Function: {result['cost']:.4f}")
    
except ImportError:
    print("Quantum module not available (Qiskit not installed)")
    print("\nSimulated QAOA Result:")
    print(f"  Selected Assets: ['AAPL', 'MSFT', 'NVDA']")
    print(f"  Expected Return: 17.0%")
    print(f"  Expected Risk: 22.3%")


# =============================================================================
# 6. OPTIONS VOLATILITY SURFACE
# =============================================================================

print("\n\n  VOLATILITY SURFACE ANALYSIS")
print("-" * 60)

from giga_system.strategies import VolatilitySurface

# Create volatility surface
strikes = np.array([80, 90, 95, 100, 105, 110, 120])
maturities = np.array([7, 14, 30, 60, 90, 180])  # Days

# Simulated market IVs (with skew)
spot = 100
base_vol = 0.20

market_ivs = np.zeros((len(maturities), len(strikes)))
for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        moneyness = np.log(K / spot)
        
        # Skew: OTM puts have higher IV
        skew = -0.1 * moneyness
        
        # Term structure: vol term structure
        term = 0.02 * np.log(T / 30)
        
        # Smile: ATM has lowest vol
        smile = 0.05 * moneyness ** 2
        
        market_ivs[i, j] = base_vol + skew + term + smile

print("Implied Volatility Surface:")
print("-" * 50)
print(f"{'Strike':<10}", end='')
for T in maturities:
    print(f"{T:>8}d", end='')
print()

for j, K in enumerate(strikes):
    print(f"{K:<10}", end='')
    for i in range(len(maturities)):
        print(f"{market_ivs[i,j]*100:>8.1f}%", end='')
    print()

# Calculate ATM vol term structure
atm_idx = list(strikes).index(100)
print(f"\nATM Volatility Term Structure:")
for i, T in enumerate(maturities):
    print(f"  {T:3d} days: {market_ivs[i, atm_idx]*100:.1f}%")

# Volatility smile for 30-day options
mat_idx = list(maturities).index(30)
print(f"\n30-Day Volatility Smile:")
for j, K in enumerate(strikes):
    moneyness = np.log(K / spot) * 100
    print(f"  K={K:3d} (log-m={moneyness:+5.1f}%): {market_ivs[mat_idx, j]*100:.1f}%")


# =============================================================================
# 7. DELTA HEDGING SIMULATION
# =============================================================================

print("\n\n  DELTA HEDGING SIMULATION")
print("-" * 60)

from giga_system.strategies import DeltaHedger
from giga_system.core import BlackScholesModel

# Initial position: Short 100 call options
n_options = -100
S0 = 100
K = 100
T = 30/365  # 30 days
r = 0.05
sigma = 0.20

# Initialize hedger
hedger = DeltaHedger(
    rebalance_frequency='daily',
    hedge_threshold=0.01
)

# Simulate price path
np.random.seed(123)
n_days = 30
dt = 1/365

prices = [S0]
for _ in range(n_days):
    dS = prices[-1] * (r * dt + sigma * np.sqrt(dt) * np.random.randn())
    prices.append(prices[-1] + dS)

prices = np.array(prices)
times = np.linspace(T, 0, n_days + 1)

# Calculate P&L from hedging
option_pnl = 0
hedge_pnl = 0
prev_delta = 0

print("Delta Hedging P&L Breakdown:")
print("-" * 50)
print(f"{'Day':<6}{'Spot':>10}{'Delta':>10}{'Hedge':>12}{'Option':>12}")

for i, (t, S) in enumerate(zip(times, prices)):
    if t <= 0:
        # At expiry
        payoff = max(S - K, 0)
        option_value = payoff
        delta = 1 if S > K else 0
    else:
        bs = BlackScholesModel(S=S, K=K, T=t, r=r, sigma=sigma)
        option_value = bs.call_price()
        delta = bs.delta('call')
    
    if i > 0:
        # Option P&L (we are short)
        option_pnl += n_options * (option_value - prev_option)
        
        # Hedge P&L (we are long delta shares)
        hedge_pnl += prev_delta * n_options * (S - prices[i-1])
    
    prev_option = option_value
    prev_delta = delta
    
    if i % 5 == 0 or i == n_days:
        print(f"{i:<6}{S:>10.2f}{delta:>10.4f}{hedge_pnl:>12.2f}{option_pnl:>12.2f}")

total_pnl = option_pnl + hedge_pnl
print("-" * 50)
print(f"{'Total':<6}{'':<10}{'':<10}{hedge_pnl:>12.2f}{option_pnl:>12.2f}")
print(f"\nNet P&L: ${total_pnl:.2f}")
print(f"Hedging {'reduced' if abs(total_pnl) < abs(option_pnl) else 'increased'} risk")


# =============================================================================
# 8. BACKTEST WITH REALISTIC EXECUTION
# =============================================================================

print("\n\n ️ BACKTEST WITH REALISTIC EXECUTION")
print("-" * 60)

from giga_system.backtesting import BacktestEngine, PerformanceAnalyzer

# Generate test data
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
np.random.seed(42)

# Simulate trending market
returns = np.random.randn(len(dates)) * 0.015 + 0.0003
cumret = np.cumprod(1 + returns)
prices_bt = pd.DataFrame({
    'open': 100 * cumret,
    'high': 100 * cumret * (1 + np.abs(np.random.randn(len(dates)) * 0.005)),
    'low': 100 * cumret * (1 - np.abs(np.random.randn(len(dates)) * 0.005)),
    'close': 100 * cumret,
    'volume': np.random.randint(1_000_000, 10_000_000, len(dates))
}, index=dates)

# Simple momentum strategy backtest
class SimpleMomentum:
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def generate_signals(self, data):
        signals = []
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        
        for i in range(self.lookback, len(data)):
            if momentum.iloc[i] > 0.001:
                direction = 1
            elif momentum.iloc[i] < -0.001:
                direction = -1
            else:
                direction = 0
            
            signals.append({
                'timestamp': data.index[i],
                'direction': direction,
                'strength': abs(momentum.iloc[i]) * 100
            })
        
        return signals

# Run backtest
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 10 bps
    slippage=0.0005    # 5 bps
)

strategy = SimpleMomentum(lookback=20)
signals = strategy.generate_signals(prices_bt)

# Simulate execution
equity = [100000]
position = 0

for signal in signals[:100]:  # First 100 signals
    if signal['direction'] != position:
        # Trade
        trade_cost = equity[-1] * (0.001 + 0.0005)  # Commission + slippage
        
        # Simple P&L
        idx = prices_bt.index.get_loc(signal['timestamp'])
        if idx + 1 < len(prices_bt):
            ret = (prices_bt['close'].iloc[idx + 1] / prices_bt['close'].iloc[idx] - 1)
            pnl = signal['direction'] * ret * equity[-1] - trade_cost
            equity.append(equity[-1] + pnl)
        
        position = signal['direction']

equity = np.array(equity)

# Calculate metrics
total_return = (equity[-1] - equity[0]) / equity[0]
ann_return = total_return * 252 / 100
ann_vol = np.std(np.diff(equity) / equity[:-1]) * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0
max_dd = np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))

print("Backtest Results:")
print("-" * 40)
print(f"  Initial Capital:  ${equity[0]:,.0f}")
print(f"  Final Capital:    ${equity[-1]:,.0f}")
print(f"  Total Return:     {total_return*100:.1f}%")
print(f"  Annual Return:    {ann_return*100:.1f}%")
print(f"  Annual Vol:       {ann_vol*100:.1f}%")
print(f"  Sharpe Ratio:     {sharpe:.2f}")
print(f"  Max Drawdown:     {max_dd*100:.1f}%")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n\n" + "=" * 70)
print("  Advanced Example Complete!")
print("=" * 70)
print("""
Features demonstrated:
1.   Pairs Trading with Cointegration Analysis
2.   GARCH Volatility Modeling (R Analytics)  
3.   Hidden Markov Model Regime Detection
4.   Mean-Variance Portfolio Optimization
5.   Quantum Portfolio Optimization (QAOA)
6.   Implied Volatility Surface Analysis
7.   Delta Hedging Simulation
8.   Backtesting with Realistic Execution

Next steps:
- Explore the Streamlit dashboard for interactive analysis
- Customize strategies for your specific use case
- Connect to real market data sources
- Deploy with Docker for production use
""")
