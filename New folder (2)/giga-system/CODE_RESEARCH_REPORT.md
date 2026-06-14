# GIGA-SYSTEM: Comprehensive Code Research Report

> **Generated**: Exhaustive line-by-line analysis of every Python file across 10 directories  
> **Total Files Analyzed**: ~69 Python files  
> **Estimated Total Lines of Code**: ~30,000+

---

## Table of Contents

1. [research/core/ (22 files)](#1-researchcore-22-files)
2. [research/ml/ (4 files)](#2-researchml-4-files)
3. [research/strategies/ (7 files)](#3-researchstrategies-7-files)
4. [research/quantum/ (7 files)](#4-researchquantum-7-files)
5. [data/ + data/live/ (10 files)](#5-data--datalive-10-files)
6. [bridge/ (7 files)](#6-bridge-7-files)
7. [execution/ (8 files)](#7-execution-8-files)
8. [risk/ (2 files)](#8-risk-2-files)
9. [account/ (1 file)](#9-account-1-file)
10. [brain/ (1 file)](#10-brain-1-file)

---

## 1. research/core/ (22 files)

### 1.1 `__init__.py` (~90 lines)

**Key Imports/Exports**: Central export hub for all core modules.

**Exports**: `BlackScholesModel`, `BinomialTree`, `MonteCarloEngine`, `StochasticModels`, `VolatilitySurface`, `ImpliedVolatility`, `GreeksCalculator`, `MarketStateSpace`, `AlphaSignalEngine`, `AlphaFactorLibrary`, `CrossSectionalAlpha`, `MicrostructureAlpha`, `InformationGeometry`, `TimeAsymmetry`, `GreekMathematics`, `RiskMetrics`, `GreeksHedging`, `GreekWalkForward`, `GreekResponse`, `DomainDataConnector`, `OptionsDataFeed`.

---

### 1.2 `alpha_factor_library.py` (~485 lines)

**Key Imports**: `numpy`, `pandas`, `scipy.stats`

**Classes**:

#### `AlphaFactorLibrary`
- **`__init__(self, lookback_period=252, risk_free_rate=0.02)`**
- **Public Methods**:
  - `momentum_factor(prices, periods=[1,5,21,63,126,252])` → `Dict[str, ndarray]` — Multi-period momentum using log returns
  - `mean_reversion_factor(prices, period=21)` → `Dict[str, ndarray]` — Z-score based: `z = (price - μ) / σ`
  - `volatility_factor(returns, windows=[5,21,63])` → `Dict[str, ndarray]` — Rolling realized vol, vol-of-vol, vol skew
  - `microstructure_factor(high, low, close, volume)` → `Dict[str, ndarray]` — Amihud illiquidity `|r|/V`, Roll spread `2*sqrt(-cov(Δp_t,Δp_{t-1}))`, Kyle's lambda `Cov(Δp,V)/Var(V)`
  - `statistical_factor(returns)` → `Dict[str, ndarray]` — Skewness, Kurtosis, Jarque-Bera, Hurst exponent
  - `cross_sectional_factor(prices_matrix, returns_matrix)` → `Dict[str, ndarray]` — Beta, idiosyncratic vol, relative momentum
  - `compute_all_factors(prices, returns, high, low, volume)` → `Dict` — Computes all factor categories

**Constants/Thresholds**: Lookback 252 (1 year), Hurst exponent via R/S analysis, JB test for normality.

---

### 1.3 `alpha_signal_engine.py` (~610 lines)

**Key Imports**: `numpy`, `scipy.stats`, `dataclasses`

**Dataclasses**:
- `AlphaSignal(name, value, confidence, timestamp, metadata, signal_type, half_life)`
- `SignalCombination(signals, combined_value, method, weights)`

**Classes**:

#### `AlphaSignalEngine`
- **`__init__(self, lookback=252, signal_decay=0.95, min_confidence=0.6)`**
- **Public Methods**:
  - `generate_momentum_signal(returns, short=5, medium=21, long=63)` → `AlphaSignal` — Weighted momentum `0.5*short + 0.3*medium + 0.2*long`, normalized, confidence from t-stat
  - `generate_mean_reversion_signal(prices, period=21)` → `AlphaSignal` — Z-score reversal, confidence = `min(|z|/3, 1.0)`
  - `generate_volatility_signal(returns, short_window=5, long_window=63)` → `AlphaSignal` — Vol ratio `short/long`, signal `1 - ratio`, confidence from Bartlett test
  - `generate_options_signal(iv, hv, delta, gamma, theta)` → `AlphaSignal` — IV premium `(IV-HV)/HV`, Greeks-weighted composite
  - `generate_microstructure_signal(bid, ask, volume, trade_flow)` → `AlphaSignal` — Spread Z-score, volume anomaly, OFI
  - `combine_signals(signals, method='weighted_average')` → `SignalCombination` — Methods: weighted_average, rank, ic_weighted
  - `decay_signals(signals, current_time)` → `List[AlphaSignal]` — Exponential decay `value * decay^elapsed`

---

### 1.4 `black_scholes.py` (~510 lines)

**Key Imports**: `numpy`, `scipy.stats`, `numba` (JIT)

**JIT Functions** (all `@jit(nopython=True, cache=True)`):
- `_norm_cdf(x)` — Abramowitz & Stegun approximation, P<1.5×10⁻⁷
- `_norm_pdf(x)` — Standard normal PDF
- `_bs_d1(S, K, T, r, sigma)` → `(ln(S/K) + (r + σ²/2)T) / (σ√T)`
- `_bs_d2(d1, sigma, T)` → `d1 - σ√T`
- `_bs_call_price(S, K, T, r, sigma)` → `S·N(d1) - K·e^{-rT}·N(d2)`
- `_bs_put_price(S, K, T, r, sigma)` → `K·e^{-rT}·N(-d2) - S·N(-d1)`

**Classes**:

#### `BlackScholesModel`
- **`__init__(self, risk_free_rate=0.05)`**
- **Public Methods**:
  - `price(S, K, T, sigma, option_type='call')` → `float` — Dispatches to JIT call/put
  - `delta(S, K, T, sigma, option_type)` → `float` — Call: `N(d1)`, Put: `N(d1)-1`
  - `gamma(S, K, T, sigma)` → `float` — `n(d1)/(S·σ·√T)`
  - `theta(S, K, T, sigma, option_type)` → `float` — Time decay per year
  - `vega(S, K, T, sigma)` → `float` — `S·n(d1)·√T`
  - `rho(S, K, T, sigma, option_type)` → `float`
  - `all_greeks(S, K, T, sigma, option_type)` → `Dict` — All 5 Greeks in one call
  - `implied_volatility(price, S, K, T, option_type, tol=1e-8, max_iter=100)` → `float` — Newton-Raphson with vega bisection fallback

**Performance Target**: `<0.001ms` per price, `<0.005ms` for all 5 Greeks.

---

### 1.5 `binomial_tree.py` (~420 lines)

**Key Imports**: `numpy`, `numba`

**Classes**:

#### `BinomialTree`
- **`__init__(self, n_steps=100, model='crr')`**
- **Models**: CRR (Cox-Ross-Rubinstein), JR (Jarrow-Rudd), Tian
- **Public Methods**:
  - `price(S, K, T, r, sigma, option_type='call', exercise='european')` → `float` — Forward tree construction + backward induction
  - `price_american(S, K, T, r, sigma, option_type)` → `float` — Early exercise at each node: `max(holding_value, intrinsic_value)`
  - `greeks(S, K, T, r, sigma, option_type, exercise)` → `Dict` — Finite difference Greeks using tree perturbation
  - `early_exercise_boundary(…)` → `ndarray` — Critical stock prices where early exercise is optimal

**Formulas**:
- CRR: `u = e^{σ√Δt}`, `d = 1/u`, `p = (e^{rΔt} - d)/(u - d)`
- JR: `u = e^{(r-σ²/2)Δt + σ√Δt}`, `d = e^{(r-σ²/2)Δt - σ√Δt}`, `p = 0.5`
- Tian: Third-moment matching model

---

### 1.6 `cross_sectional_alpha.py` (~480 lines)

**Key Imports**: `numpy`, `scipy.stats`, `sklearn.linear_model`

**Classes**:

#### `CrossSectionalAlpha`
- **`__init__(self, n_quantiles=5, holding_period=21, risk_free_rate=0.02)`**
- **Public Methods**:
  - `rank_assets(factor_values, method='percentile')` → `ndarray` — Percentile or Z-score ranking
  - `long_short_portfolio(returns, factor_values, n_quantiles=5)` → `Dict` — Top quintile long, bottom quintile short, equal-weight
  - `factor_ic(factor_values, forward_returns)` → `Dict` — Information Coefficient: Spearman rank corr + t-stat
  - `factor_premium(returns, factor_values, method='fama_macbeth')` → `Dict` — Fama-MacBeth cross-sectional regression: `R_i = α + β·Factor_i + ε`, NW standard errors
  - `factor_decay(factor_values, forward_returns, horizons)` → `Dict` — IC at multiple horizons, optimal horizon detection
  - `turnover_analysis(factor_values_series)` → `Dict` — Portfolio turnover, quintile transition matrix

---

### 1.7 `domain_data_connector.py` (~390 lines)

**Key Imports**: `numpy`, `dataclasses`

**5-Domain Greek Alpha Framework**:
- **State Space (Ω)**: Market regimes — `{bull, bear, neutral, volatile, crisis}`
- **Variational Sensitivity (Δ)**: Delta/gamma exposure, position Greeks
- **Stochastic Motion (κ)**: Volatility dynamics, mean reversion speed
- **Ergodicity (τ)**: Time-reversibility, stationarity tests
- **Information Geometry (Η)**: Fisher information, KL divergence, entropy

**Dataclasses**: `DomainState` (per domain), `DomainSnapshot` (all 5 domains at once)

**Classes**:

#### `DomainDataConnector`
- **`__init__(self, kappa_threshold=0.5, entropy_threshold=2.0)`**
- **Public Methods**:
  - `compute_omega(prices, returns)` → `DomainState` — Regime via trend + vol clustering
  - `compute_delta(greeks_dict)` → `DomainState` — Net delta/gamma/theta, $ exposure
  - `compute_kappa(returns, window=63)` → `DomainState` — Realized vol, vol-of-vol, mean reversion speed (OU estimate)
  - `compute_tau(returns)` → `DomainState` — ADF stationary test, Hurst exponent, time-reversal asymmetry
  - `compute_eta(returns)` → `DomainState` — Fisher information, Shannon entropy, KL divergence from normal
  - `full_snapshot(prices, returns, greeks_dict)` → `DomainSnapshot`
  - `alpha_opportunity(snapshot)` → `Dict` — **Central Hypothesis**: High κ + Low Entropy = Max Alpha

---

### 1.8 `greeks.py` (~570 lines)

**Key Imports**: `numpy`, `numba`, `scipy.stats`

**JIT Functions**: `_fast_delta`, `_fast_gamma`, `_fast_theta`, `_fast_vega`, `_fast_rho`, `_fast_all_greeks` — All Numba-compiled for HFT speed.

**Classes**:

#### `GreeksCalculator`
- **`__init__(self, model='black_scholes')`**
- **Public Methods**:
  - `calculate(S, K, T, r, sigma, option_type)` → `Dict` — All 5 Greeks via JIT
  - `delta(…)`, `gamma(…)`, `theta(…)`, `vega(…)`, `rho(…)` — Individual Greeks
  - `portfolio_greeks(positions: List[Dict])` → `Dict` — Net portfolio Greeks with $ exposure
  - `greek_sensitivities(S, K, T, r, sigma)` → `Dict` — Second-order Greeks: vanna, volga (vomma), charm, color, speed, zomma, ultima
  - `finite_difference_greeks(pricing_func, S, K, T, r, sigma)` → `Dict` — Numerical Greeks via central differences

**Second-Order Greek Formulas**:
- Vanna: `∂²V/∂S∂σ = -n(d1)·d2/σ`
- Volga (Vomma): `∂²V/∂σ² = vega·d1·d2/σ`
- Charm: `∂²V/∂S∂t = -n(d1)·(2rT-d2·σ√T)/(2T·σ√T)`
- Speed: `∂³V/∂S³ = -(gamma/S)·(d1/(σ√T)+1)`

---

### 1.9 `greeks_hedging.py` (~540 lines)

**Key Imports**: `numpy`, `scipy.optimize`

**Classes**:

#### `GreeksHedging`
- **`__init__(self, risk_free_rate=0.05, transaction_cost_bps=5)`**
- **Public Methods**:
  - `delta_hedge(portfolio_delta, S, hedge_instrument_delta=1.0)` → `Dict` — Hedge ratio = `-Δ_portfolio/Δ_instrument`
  - `delta_gamma_hedge(delta, gamma, instruments)` → `Dict` — Simultaneous delta+gamma neutralization via 2-instrument linear system
  - `minimum_variance_hedge(portfolio_greeks, instruments, target_greeks)` → `Dict` — Quadratic programming: `min ||w·G - target||² + λ·Σ(|w|·tc)`
  - `dynamic_hedge_simulation(S0, K, T, sigma, n_rebalances)` → `Dict` — Simulated hedging P&L with discrete rebalancing
  - `optimal_rebalance_frequency(S, K, T, sigma, tc_bps)` → `Dict` — Optimal rebalance period: `Δt* = (3·tc/(2·γ·σ²))^{2/3}`, Leland-adjusted vol: `σ̃ = σ·√(1 + √(2/π)·tc/(σ·√Δt))`
  - `hedge_ratio_sensitivity(S, K, T, sigma)` → `Dict` — Delta sensitivity to S, σ, T changes

---

### 1.10 `greek_mathematics.py` (~600 lines)

**Key Imports**: `numpy`, `scipy`, `math`

**10 Classes representing ancient Greek mathematical concepts applied to trading**:

1. `PythagoreanHarmony` — Harmonic price ratios, Pythagorean triples in returns
2. `EuclideanGeometry` — GCD-based support/resistance, Euclidean distance metrics
3. `ArchimedeanSpirals` — Log-spiral price targets, Archimedes' approximation of π
4. `FibonacciPatterns` — Golden ratio (φ=1.618), Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
5. `PlatonicGeometry` — Platonic solid symmetries in portfolio allocation
6. `AristotleLogic` — Syllogistic reasoning for market state classification
7. `ThalesTheorem` — Geometric mean, Thales' intercept theorem price targets
8. `HipparchiusAstronomy` — Celestial cycle analysis (lunar 29.5d, solar 365.25d, Mercury 88d)
9. `DiophantineEquations` — Integer ratio analysis, rational approximations
10. `ZenoParadox` — Convergence analysis, Zeno-style limit computations

---

### 1.11 `greek_response.py` (~440 lines)

**Key Imports**: `numpy`, `dataclasses`

**Dataclass**: `GreekResponse(delta, gamma, theta, vega, rho, timestamp, source, metadata)`

**Classes**:

#### `GreekResponseAnalyzer`
- **`__init__(self, lookback=100, sensitivity_threshold=0.05)`**
- **Public Methods**:
  - `analyze_response(greek_history: List[GreekResponse])` → `Dict` — Time-series analysis of Greeks: trends, volatility, anomalies
  - `detect_greek_anomalies(responses, z_threshold=2.5)` → `List[Dict]` — Z-score anomaly detection
  - `greek_contribution_analysis(response, price_change, vol_change, time_change)` → `Dict` — P&L attribution: `ΔV ≈ Δ·ΔS + ½Γ·ΔS² + Θ·Δt + ν·Δσ`
  - `regime_greek_profile(responses, regime_labels)` → `Dict` — Average Greeks per market regime

---

### 1.12 `greek_walk_forward.py` (~460 lines)

**Key Imports**: `numpy`, `pandas`

**Classes**:

#### `GreekWalkForward`
- **`__init__(self, train_period=252, test_period=21, n_splits=12)`**
- **Public Methods**:
  - `walk_forward_test(returns, factor_values, strategy_func)` → `Dict` — Walk-forward optimization with IS/OOS performance tracking
  - `expanding_window_test(returns, factor_values, strategy_func, min_train=126)` → `Dict` — Expanding training window
  - `calculate_oos_metrics(oos_returns)` → `Dict` — Sharpe, Sortino, max DD, Calmar, win rate, profit factor
  - `detect_overfitting(is_performance, oos_performance)` → `Dict` — Overfitting metrics: IS/OOS Sharpe ratio, performance decay

---

### 1.13 `implied_volatility.py` (~480 lines)

**Key Imports**: `numpy`, `numba`, `scipy.stats`

**JIT Functions**: `_iv_newton_step`, `_iv_bisection_step`

**Classes**:

#### `ImpliedVolatility`
- **`__init__(self, tol=1e-8, max_iter=100)`**
- **Public Methods**:
  - `calculate(option_price, S, K, T, r, option_type)` → `float` — Hybrid Newton-Raphson + bisection: initial guess from Brenner-Subrahmanyam `σ₀ ≈ √(2π/T) · price/S`
  - `iv_surface(prices_matrix, strikes, expiries, S, r)` → `ndarray` — Full IV surface computation
  - `calculate_smile(option_prices, strikes, S, T, r)` → `Dict` — Smile metrics: ATM vol, skew `(IV_90 - IV_110)/IV_ATM`, kurtosis (butterfly)
  - `term_structure(atm_prices, expiries, S, r)` → `Dict` — IV term structure analysis

**Performance Target**: `<0.1ms` per IV calculation.

---

### 1.14 `information_geometry.py` (~530 lines)

**Key Imports**: `numpy`, `scipy.stats`, `scipy.linalg`

**Classes**:

#### `InformationGeometry`
- **`__init__(self, distribution='normal')`**
- **Public Methods**:
  - `fisher_information_matrix(params, distribution)` → `ndarray` — FIM for normal/student-t/log-normal
  - `natural_gradient(params, loss_gradient, fim)` → `ndarray` — `∇̃ = F⁻¹ · ∇`, with regularization `(F + εI)⁻¹`
  - `geodesic_distance(params1, params2)` → `float` — Distance on statistical manifold: `d = √(Δθᵀ·F·Δθ)`
  - `kl_divergence(params1, params2, distribution)` → `float` — KL(P||Q) for various distributions
  - `entropy(params, distribution)` → `float` — Differential entropy: Normal: `½ln(2πeσ²)`
  - `market_complexity(returns)` → `Dict` — Fisher info, entropy, vol-of-vol, complexity score
  - `regime_distance(regime1_params, regime2_params)` → `Dict` — KL divergence + geodesic between regimes

---

### 1.15 `market_state_space.py` (~500 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Enums**: `MarketRegime(TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, CRISIS)`

**Classes**:

#### `MarketStateSpace`
- **`__init__(self, n_regimes=4, lookback=252)`**
- **Public Methods**:
  - `estimate_state(returns)` → `Dict` — GMM-based regime classification with transition probabilities
  - `transition_matrix(regime_history)` → `ndarray` — Empirical Markov transition matrix
  - `stationary_distribution(transition_matrix)` → `ndarray` — Eigenvector method: find π where π·P = π
  - `expected_regime_duration(transition_matrix)` → `ndarray` — `E[duration_i] = 1/(1-p_ii)`
  - `regime_conditional_stats(returns, regimes)` → `Dict` — Per-regime mean, vol, Sharpe, skewness, kurtosis
  - `hidden_markov_model(returns, n_states)` → `Dict` — Custom EM algorithm for HMM
  - `detect_regime_change(returns, window, threshold)` → `Dict` — CUSUM-based change detection

---

### 1.16 `microstructure_alpha.py` (~510 lines)

**Key Imports**: `numpy`, `numba`

**JIT Functions**: `_fast_kyle_lambda`, `_fast_roll_spread`, `_fast_amihud`

**Classes**:

#### `MicrostructureAlpha`
- **`__init__(self, tick_size=0.01)`**
- **Public Methods**:
  - `effective_spread(trade_prices, midpoints)` → `ndarray` — `2·|P_trade - P_mid|`
  - `realized_spread(trade_prices, midpoints, delay=5)` → `ndarray` — With delay: `2·D·(P_trade - P_{mid+delay})`
  - `price_impact(trade_prices, volumes, direction)` → `Dict` — Kyle's lambda, Amihud, permanent/temporary impact decomposition
  - `order_flow_imbalance(buy_volume, sell_volume, window=20)` → `Dict` — OFI, VPIN (Volume-synchronized PIN)
  - `information_share(prices_venue1, prices_venue2)` → `Dict` — Hasbrouck information shares via VECM
  - `adverse_selection(trade_prices, midpoints, volumes)` → `Dict` — PIN model, adverse selection cost
  - `toxicity_index(prices, volumes)` → `Dict` — VPIN toxic flow index

**Formulas**:
- Kyle's Lambda: `λ = Cov(Δp, signed_volume) / Var(signed_volume)`
- Roll Spread: `S = 2·√(-Cov(Δp_t, Δp_{t-1}))`
- Amihud: `ILLIQ = (1/N)·Σ|r_t|/V_t`

---

### 1.17 `monte_carlo.py` (~580 lines)

**Key Imports**: `numpy`, `numba`, `scipy.stats`

**Classes**:

#### `MonteCarloEngine`
- **`__init__(self, n_simulations=100000, n_steps=252, seed=None)`**
- **Public Methods**:
  - `simulate_gbm(S0, mu, sigma, T)` → `ndarray` — Geometric Brownian Motion: `S_t = S_0·exp((μ-σ²/2)t + σW_t)`, uses antithetic variates
  - `simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, T)` → `Tuple` — Full Heston SV model with Milstein discretization, absorption for v<0
  - `price_european(S0, K, T, r, sigma, option_type)` → `Dict` — GBM-based MC pricing with SE, 95% CI
  - `price_american(S0, K, T, r, sigma, option_type, n_basis=5)` → `Dict` — Longstaff-Schwartz LSM with Laguerre polynomial basis
  - `price_exotic(S0, K, T, r, sigma, payoff_func)` → `Dict` — Custom payoff MC pricing
  - `calculate_var(portfolio_returns, confidence, horizon)` → `Dict` — Historical + parametric VaR/CVaR
  - `importance_sampling(S0, K, T, r, sigma, option_type, shift)` → `Dict` — Drift shift for OTM options: `importance_weight = exp(-shift·Z - shift²/2)`

**Variance Reduction**: Antithetic variates (all simulations), importance sampling (OTM pricing).

---

### 1.18 `options_data_feed.py` (~420 lines)

**Key Imports**: `numpy`, `dataclasses`, `requests`, `yfinance`

**Dataclasses**: `OptionQuote(symbol, expiry, strike, option_type, bid, ask, last, volume, oi, iv, delta, gamma, theta, vega, underlying_price, timestamp)`

**Classes**:

#### `OptionsDataFeed`
- **`__init__(self, data_source='yahoo')`**
- **Public Methods**:
  - `get_option_chain(symbol, expiry)` → `List[OptionQuote]` — Via yfinance or synthetic
  - `get_all_expirations(symbol)` → `List` — Available expiry dates
  - `_generate_synthetic_chain(symbol, S, expiries)` → `List[OptionQuote]` — BS-model-based synthetic chain with realistic spread modeling

---

### 1.19 `risk_metrics.py` (~520 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `RiskMetrics`
- **`__init__(self, risk_free_rate=0.02, annualization=252)`**
- **Public Methods**:
  - `sharpe_ratio(returns)` → `float` — `(E[r]-r_f)/σ · √252`
  - `sortino_ratio(returns, target=0)` → `float` — Uses downside deviation only
  - `calmar_ratio(returns)` → `float` — `Ann. Return / Max Drawdown`
  - `max_drawdown(returns)` → `Dict` — Peak-to-trough + recovery analysis
  - `value_at_risk(returns, confidence, method='historical')` → `Dict` — Historical/Parametric/Cornish-Fisher VaR
  - `conditional_var(returns, confidence)` → `float` — CVaR = E[R | R < VaR]
  - `omega_ratio(returns, threshold=0)` → `float` — Area above threshold / area below
  - `tail_ratio(returns)` → `float` — `P95/|P05|`
  - `portfolio_risk_decomposition(weights, cov_matrix)` → `Dict` — Marginal/component risk contributions
  - `stress_test(returns, scenarios)` → `Dict` — Regime-conditional scenario analysis

**Cornish-Fisher VaR**: `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36`

---

### 1.20 `stochastic_models.py` (~720 lines)

**Key Imports**: `numpy`, `numba`, `scipy.optimize`

**Classes**:

#### `StochasticModels`
- **`__init__(self)`**
- **Public Methods**:
  - `ornstein_uhlenbeck(X0, theta, mu, sigma, T, n_steps)` → `Tuple` — `dX = θ(μ-X)dt + σdW`, Euler-Maruyama
  - `heston_model(S0, v0, r, kappa, theta, sigma_v, rho, T)` → `Tuple` — Full Heston with correlation: `dS = rSdt + √vSdW₁`, `dv = κ(θ-v)dt + σ_v√vdW₂`, Feller condition: `2κθ > σ_v²`
  - `jump_diffusion(S0, mu, sigma, lam, mu_j, sigma_j, T)` → `Tuple` — Merton model: `dS/S = (μ-λk)dt + σdW + JdN`, compound Poisson
  - `sabr_model(F0, sigma0, alpha, beta, rho, T)` → `Tuple` — Stochastic Alpha Beta Rho: `dF = σF^βdW₁`, `dσ = ασdW₂`
  - `sabr_implied_vol(F, K, T, sigma0, alpha, beta, rho)` → `float` — Hagan SABR formula for approximated IV
  - `fit_ou_parameters(data)` → `Dict` — MLE fit for OU process: `θ̂ = -ln(ρ̂)/Δt`, `μ̂ = mean(X)`
  - `fit_heston_parameters(returns, prices)` → `Dict` — Method of moments + SLSQP refinement

---

### 1.21 `time_asymmetry.py` (~430 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `TimeAsymmetry`
- **`__init__(self, lookback=252)`**
- **Public Methods**:
  - `time_reversal_asymmetry(returns, lag=1)` → `Dict` — `TRA = E[r_t² · r_{t-lag}] - E[r_t · r_{t-lag}²]`, normalized by `σ³`
  - `leverage_effect(returns, volatility, lag=1)` → `Dict` — `L(τ) = Corr(r_t, |r_{t+τ}|²)`, expected negative for equities
  - `gain_loss_asymmetry(returns, threshold_quantile=0.1)` → `Dict` — Duration asymmetry: avg gain duration vs avg loss duration
  - `volatility_clustering(returns, lags)` → `Dict` — Autocorrelation of `|r_t|` at multiple lags
  - `decay_analysis(autocorrelations, lags)` → `Dict` — Fit exponential decay `A·e^{-t/τ}` to autocorrelation

---

### 1.22 `volatility_surface.py` (~550 lines)

**Key Imports**: `numpy`, `scipy.interpolate`, `scipy.optimize`

**Classes**:

#### `VolatilitySurface`
- **`__init__(self, interpolation='cubic')`**
- **Public Methods**:
  - `build_surface(strikes, expiries, iv_data, S, r)` → `Dict` — Construct 2D IV surface with moneyness transformation
  - `get_iv(K, T)` → `float` — Interpolated IV at arbitrary (K, T) via `RectBivariateSpline` or `griddata`
  - `fit_svi(strikes, ivs, T)` → `Dict` — SVI model: `w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))`, SLSQP fit with butterfly arbitrage constraint
  - `local_volatility(S, K, T)` → `float` — Dupire formula: `σ_L² = (∂C/∂T + rK·∂C/∂K) / (½K²·∂²C/∂K²)`
  - `smile_dynamics(iv_history, strikes, S_history)` → `Dict` — Sticky-strike vs sticky-delta classification, smile movement correlation
  - `surface_metrics(strikes, expiries, ivs, S)` → `Dict` — ATM vol, term structure slope, skew, butterfly, risk-reversal

---

## 2. research/ml/ (4 files)

### 2.1 `__init__.py` (~25 lines)

**Exports**: `FeatureEngine`, `RegimeDetector`, `VolatilityForecaster`

---

### 2.2 `feature_engineering.py` (~540 lines)

**Key Imports**: `numpy`, `pandas`, `scipy.stats`

**Classes**:

#### `FeatureEngine`
- **`__init__(self, lookback_periods=[5,10,21,63,126,252])`**
- **Public Methods**:
  - `create_features(prices, volumes, returns)` → `Dict[str, ndarray]` — 30+ feature families:
    - **Price**: Log-returns, momentum (multi-period), SMA ratios, Bollinger %B
    - **Volume**: Volume ratio, VWAP deviation, OBV slope
    - **Volatility**: Realized vol (multiple windows), Parkinson `σ_P = √(ln(H/L)²/(4ln2))`, Garman-Klass, Yang-Zhang, vol-of-vol
    - **Microstructure**: Amihud, Roll spread
    - **Statistical**: Skewness, Kurtosis, Hurst exponent, autocorrelation
    - **Cross-sectional**: Beta, idiosyncratic vol (if multiple assets)
  - `normalize_features(features, method='zscore')` → `Dict` — Z-score, MinMax, or rank normalization
  - `select_features(features, target, method='mutual_info', k=10)` → `Dict` — Correlation filter + mutual information ranking

---

### 2.3 `regime_detection.py` (~490 lines)

**Key Imports**: `numpy`, `sklearn.mixture`, `scipy`

**Enums**: `Regime(BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL, CRISIS)`

**Classes**:

#### `RegimeDetector`
- **`__init__(self, n_regimes=4, method='gmm')`**
- **Methods**: `gmm` or `hmm`
- **Public Methods**:
  - `fit(returns, volumes=None)` → `self` — Trains GMM with features: returns, vol, rolling Sharpe, vol-of-vol
  - `predict(returns, volumes=None)` → `ndarray` — Current regime labels
  - `transition_matrix()` → `ndarray` — Empirical transition probabilities
  - `regime_statistics(returns, regimes)` → `Dict` — Per-regime stats (mean, vol, Sharpe, drawdown)
  - `detect_change_points(returns, method='cusum', threshold=3.0)` → `List` — CUSUM: `S_t = max(0, S_{t-1} + (|r_t - μ̂| - 0.5σ̂))` or Bayesian change detection

---

### 2.4 `volatility_forecast.py` (~480 lines)

**Key Imports**: `numpy`, `scipy.optimize`

**Classes**:

#### `VolatilityForecaster`
- **`__init__(self, models=['ewma', 'garch', 'har'])`**
- **Public Methods**:
  - `fit(returns)` → `self` — Trains all specified models
  - `forecast(horizon=1)` → `Dict` — Ensemble forecast with per-model outputs
  - `_fit_ewma(returns, lambda_=0.94)` — `σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}`
  - `_fit_garch(returns)` — GARCH(1,1): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}`, MLE via SLSQP, persistence `α+β`
  - `_fit_har(returns)` — HAR-RV (Heterogeneous Autoregressive): `RV_t = β₁·RV_d + β₂·RV_w + β₃·RV_m + ε`, OLS fit
  - `_ensemble_forecast(horizon)` — Weighted average: EWMA 0.3, GARCH 0.4, HAR 0.3
  - `evaluate(returns, test_size=63)` → `Dict` — RMSE, MAE, directional accuracy

---

## 3. research/strategies/ (7 files)

### 3.1 `__init__.py` (~35 lines)

**Exports**: `Strategy`, `Signal`, `Position`, `Order`, `MarketMaker`, `TrendFollowing`, `BreakoutStrategy`, `LiveMomentumStrategy`, `DeltaHedging`, `VolatilityArbitrage`, `IronCondorStrategy`, `PairsTrading`, `AdaptiveParameters`

---

### 3.2 `base.py` (~390 lines)

**Key Imports**: `numpy`, `abc`, `dataclasses`, `enum`

**Enums**: `SignalType(LONG, SHORT, FLAT)`, `OrderType(MARKET, LIMIT, STOP)`

**Dataclasses**:
- `Signal(signal_type, strength, confidence, timestamp, metadata, expires_at, source)`
- `Position(symbol, quantity, entry_price, current_price, unrealized_pnl, realized_pnl, entry_time, max_favorable, max_adverse)`
- `Order(symbol, order_type, side, quantity, price, stop_price, take_profit, time_in_force)`

**Classes**:

#### `Strategy` (ABC)
- **`__init__(self, name, capital=1_000_000, max_position_pct=0.1, max_drawdown=0.15)`**
- **Abstract Methods**: `generate_signal(market_data)` → `Optional[Signal]`, `size_position(signal, market_data)` → `Optional[Order]`
- **Concrete Methods**:
  - `update(market_data)` → `Optional[Order]` — Main loop: signal → size → track
  - `kelly_criterion(win_prob, win_loss_ratio)` → `float` — `f* = p - (1-p)/b`
  - `calculate_metrics()` → `Dict` — Sharpe, max DD, win rate, profit factor, expectancy

---

### 3.3 `market_making.py` (~520 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `MarketMaker`
- **`__init__(self, symbol, capital, tick_size=0.01, max_position=1000, target_spread_bps=5, inventory_skew=0.5, max_inventory_risk=0.02)`**
- **Public Methods**:
  - `generate_signal(market_data)` → `Signal` — Avellaneda-Stoikov framework
  - `_calculate_optimal_spread(S, sigma, gamma, dt)` → `float` — `spread = γσ²(T-t) + (2/γ)·ln(1+γ/κ)`, where κ is market order arrival rate
  - `_inventory_adjustment(current_inventory, max_inventory)` → `float` — Linear inventory skew
  - `_quote_prices(mid, spread, skew)` → `Tuple[float,float]` — `bid = mid - spread/2 - skew`, `ask = mid + spread/2 - skew`
  - `calculate_pnl(fills)` → `Dict` — Realized spread, inventory P&L, total P&L

**Formula**: Avellaneda-Stoikov reservation price: `r = S - q·γ·σ²·(T-t)`

---

### 3.4 `momentum.py` (~580 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `TrendFollowing`
- **`__init__(self, name='trend_following', fast_period=10, slow_period=50, atr_period=14, risk_per_trade=0.02)`**
- **Signal Logic**: Dual EMA crossover with ATR-based position sizing
- **Position Sizing**: `quantity = (capital · risk_per_trade) / (atr_multiplier · ATR)`

#### `BreakoutStrategy`
- **`__init__(self, name='breakout', lookback=20, volume_threshold=1.5, atr_period=14)`**
- **Signal Logic**: Donchian channel breakout with volume confirmation

#### `LiveMomentumStrategy`
- **`__init__(self, symbol='BTCUSDT', lookback=20, atr_period=14, risk_fraction=0.01)`**
- **Signal Logic**: EMA crossover + ATR volatility filter + RSI filter, designed for live crypto trading

---

### 3.5 `options_strategies.py` (~520 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `DeltaHedging`
- **`__init__(self, rebalance_threshold=0.05, hedge_ratio=1.0)`**
- **Signal Logic**: Rebalance when |accumulated_delta| > threshold

#### `VolatilityArbitrage`
- **`__init__(self, iv_hv_threshold=0.1, min_premium=0.02, lookback=21)`**
- **Signal Logic**: IV premium `(IV-HV)/HV`, sell when premium > threshold, buy when discount

#### `IronCondorStrategy`
- **`__init__(self, wing_width=0.05, premium_target=0.02, max_loss_multiple=2.0, delta_threshold=0.3)`**
- **Signal Logic**: Low-vol regime iron condor deployment, exits on delta breach

---

### 3.6 `pairs_trading.py` (~480 lines)

**Key Imports**: `numpy`, `scipy.stats`, `sklearn.linear_model`

**Classes**:

#### `PairsTrading`
- **`__init__(self, entry_z=2.0, exit_z=0.5, lookback=60, min_correlation=0.7, max_half_life=30)`**
- **Public Methods**:
  - `find_pairs(prices_dict, min_history=252)` → `List[Dict]` — ADF cointegration test on all pairs, half-life filter
  - `generate_signal(market_data)` → `Signal` — Z-score of spread: long spread if z < -entry_z, short if z > entry_z
  - `_calculate_half_life(spread)` → `float` — OU half-life: `hl = -ln(2)/β` from `Δspread = α + β·spread + ε`
  - `_calculate_hedge_ratio(y, x)` → `float` — OLS regression, dynamic re-estimation

---

### 3.7 `adaptive_params.py` (~400 lines)

**Key Imports**: `numpy`, `scipy.stats`

**Classes**:

#### `AdaptiveParameters`
- **`__init__(self, base_params, adaptation_speed=0.1, regime_sensitivity=0.5, vol_adjustment=True)`**
- **Public Methods**:
  - `update(market_data, performance)` → `Dict` — Online parameter adjustment based on recent performance and market regime
  - `_regime_adjustment(regime, base_params)` → `Dict` — Regime-conditional multipliers (e.g., widen stops in high vol)
  - `_performance_adjustment(recent_trades)` → `Dict` — Kelly-criterion-based sizing adjustment
  - `_volatility_adjustment(current_vol, historical_vol)` → `Dict` — Vol-scaled parameters

---

## 4. research/quantum/ (7 files)

### 4.1 `__init__.py` (~42 lines)

**Exports**: `QuadraticProgram`, `QuantumOptimizer`, `OptimizerType`, `OptimizationResult`, `QISKIT_AVAILABLE`, `QuantumPortfolioOptimizer`, `QuantumPortfolioResult`, `PortfolioConstraints`, `QuantumRiskAnalyzer`, `QuantumAmplitudeEstimation`, `RiskMetrics`, `ScenarioResult`

---

### 4.2 `quantum_monte_carlo.py` (~706 lines)

**Key Imports**: `numpy`, `qiskit` (QuantumCircuit, Aer, UniformDistribution, LinearAmplitudeFunction)

**Dataclass**: `QuantumEstimationResult(estimated_value, confidence_interval, estimation_error, num_oracle_calls, quantum_advantage, classical_result, algorithm_used, num_qubits, shots, execution_time_ms, backend_name)`

**Classes**:

#### `QuantumMonteCarlo`
- **`__init__(self, backend='qasm_simulator', shots=1024, error_mitigation=True)`**
- **Public Methods**:
  - `european_option_pricing(S, K, T, r, sigma, option_type='call')` → `QuantumEstimationResult` — Uses Maximum Likelihood Amplitude Estimation (MLAE) with `UniformDistribution` → `LinearAmplitudeFunction` → comparator circuit. Classical BS fallback.
  - `value_at_risk_calculation(returns, confidence=0.95)` → `QuantumEstimationResult` — Quantum indicator function for tail probability estimation
  - `_classical_option_pricing(S, K, T, r, sigma, option_type)` → `float` — Black-Scholes fallback
  - `_calculate_quantum_advantage(n_qubits, shots)` → `float` — `advantage = classical_cost / quantum_cost`, Classical: O(2^2n), Quantum: O(2^n)

#### `AmplitudeEstimation`
- **`__init__(self, n_qubits=5, backend='qasm_simulator', shots=1024)`**
- **Public Methods**:
  - `estimate_amplitude(oracle_circuit)` → `Dict` — Standalone QAE

**Convenience Functions**: `quantum_option_pricing()`, `quantum_var_calculation()`

---

### 4.3 `quantum_optimizer.py` (~569 lines)

**Key Imports**: `numpy`, `qiskit` (QuantumCircuit, Aer), `scipy.optimize` (minimize)

**Enum**: `OptimizerType(QAOA, VQE, CLASSICAL)`

**Dataclass**: `OptimizationResult(optimal_weights, optimal_value, n_iterations, eigenvalue, eigenvector, optimizer_type, execution_time, metadata)`

**Classes**:

#### `QuadraticProgram`
- **`__init__(self, n_assets: int)`**
- **QUBO**: `min x^T Q x + c^T x`
- **Public Methods**:
  - `set_portfolio_objective(expected_returns, cov_matrix, risk_aversion=1.0)` — Builds Q matrix with budget constraint penalty=10.0
  - `to_ising()` → `Tuple[ndarray, float]` — Maps binary x = (1+z)/2 to Ising Hamiltonian
  - `to_pauli_op()` → `SparsePauliOp` — Pauli Hamiltonian for quantum circuit

#### `QuantumOptimizer`
- **`__init__(self, optimizer_type=OptimizerType.QAOA, reps=3, shots=1024)`**
- **Public Methods**:
  - `optimize(program: QuadraticProgram)` → `OptimizationResult` — Dispatches to QAOA/VQE/classical
  - `_build_qaoa_circuit(gamma, beta, n_qubits)` → `QuantumCircuit` — `|ψ(β,γ)⟩ = U_B(β_p)·U_C(γ_p)...U_B(β_1)·U_C(γ_1)|+⟩^n`, RZ+RZZ(cost)+RX(mixer)
  - `_build_vqe_ansatz(n_qubits, reps)` → `QuantumCircuit` — RY + CNOT ladder
  - `_optimize_classical(program)` — Brute-force (n≤10) or SLSQP relaxation
  - `_optimize_qaoa(program)` — COBYLA parameter optimization
  - `_optimize_vqe(program)` — SLSQP parameter optimization

---

### 4.4 `risk_quantum.py` (~682 lines)

**Key Imports**: `numpy`, `scipy.stats`, `qiskit`

**Dataclasses**: `RiskMetrics(var_95, var_99, cvar_95, cvar_99, expected_shortfall, volatility, downside_deviation, max_loss, probability_of_loss, tail_ratio)`, `ScenarioResult(mean_return, volatility, var, cvar, scenarios, percentiles, distribution_params)`

**Classes**:

#### `QuantumRiskAnalyzer`
- **`__init__(self, n_qubits=8, use_quantum=True)`**
- **Public Methods**:
  - `calculate_var(returns, confidence=0.95, method='historical')` → `float` — 4 methods: historical, parametric, cornish_fisher, quantum
  - `_var_cornish_fisher(returns, confidence)` → `float` — `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36`
  - `calculate_cvar(returns, confidence=0.95)` → `float` — `CVaR = E[R | R < VaR]`
  - `monte_carlo_var(returns, n_simulations=10000, distribution='normal')` — Supports normal, t, quantum distributions
  - `_quantum_sampling(n_samples)` — Box-Muller transform with quantum random bits
  - `portfolio_var(weights, returns_matrix, confidence=0.95)` — Parametric + MC portfolio VaR
  - `stress_test(returns, scenarios)` → `Dict` — Named scenario analysis
  - `comprehensive_risk_metrics(returns)` → `RiskMetrics` — All risk measures in one call

#### `QuantumAmplitudeEstimation`
- **`__init__(self, n_qubits=5, n_iterations=3)`**
- **Public Methods**:
  - `estimate_tail_probability(returns, threshold)` → `float` — `P(R < threshold)` via quantum counting

---

### 4.5 `hybrid_algorithms.py` (~915 lines)

**Key Imports**: `numpy`, `qiskit`, `scipy.optimize`

**Dataclass**: `HybridOptimizationResult(optimal_weights, optimal_value, n_iterations, eigenvale, optimizer_type, execution_time, metadata)`

**Classes**:

#### `QuantumClassicalNeuralNetwork`
- **`__init__(self, classical_input_dim, quantum_layers=2, num_qubits=4)`**
- **Public Methods**:
  - `forward(x)` → `ndarray` — Classical NN → quantum circuit → measurement
  - `_quantum_forward(x)` — Parameterized quantum circuit evaluation
  - `train(X, y, epochs=100, lr=0.01)` — Parameter shift rule for quantum gradient: `∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2`

#### `QuantumApproximateOptimization`
- **`__init__(self, num_assets, p_layers=2, shots=1024)`**
- **Public Methods**:
  - `optimize_portfolio(expected_returns, cov_matrix, risk_aversion=1.0)` → `HybridOptimizationResult`
  - `_create_portfolio_qubo(returns, cov, risk_aversion)` — `Q = risk_aversion·Σ - (1-λ)·μ`
  - `_qubo_to_pauli(Q)` — Converts QUBO matrix to Pauli Hamiltonian
  - `_decode_solution(counts, n_assets)` — Extracts binary portfolio from measurement counts
  - `_classical_portfolio_optimization(returns, cov, risk_aversion)` — SLSQP fallback

#### `VariationalQuantumEigensolver`
- **`__init__(self, num_qubits=4)`**
- Ansatz: RealAmplitudes, Optimizer: SPSA
- **Public Methods**:
  - `find_ground_state(hamiltonian)` → `Dict` — Classical eigenvalue decomposition fallback
  - `_matrix_to_pauli(matrix)` — Expands matrix into Pauli basis

**Convenience**: `hybrid_portfolio_optimization(algorithm='qaoa'|'vqe')`

---

### 4.6 `portfolio_quantum.py` (~641 lines)

**Key Imports**: `numpy`, `scipy.optimize`, `qiskit`

**Dataclasses**: `PortfolioConstraints(min_weight=0.0, max_weight=1.0, cardinality=None, sector_limits=None, turnover_limit=None)`, `QuantumPortfolioResult(weights, expected_return, risk, sharpe_ratio, method, execution_time, quantum_advantage, metadata)`

**Classes**:

#### `QuantumPortfolioOptimizer`
- **`__init__(self, n_assets, risk_free_rate=0.02, qaoa_reps=3)`**
- **Public Methods**:
  - `optimize_mean_variance(returns, cov, risk_aversion=1.0)` → `QuantumPortfolioResult` — QUBO formulation of Markowitz
  - `compute_efficient_frontier(returns, cov, n_points=20)` → `List[QuantumPortfolioResult]` — Multiple risk aversions [0.1...10]
  - `minimum_variance(returns, cov)` → `QuantumPortfolioResult`
  - `maximum_sharpe(returns, cov)` → `QuantumPortfolioResult` — Grid search over risk aversions
  - `risk_parity(cov)` → `QuantumPortfolioResult` — Newton-Raphson iterative: equal risk contribution `w_i·(Σw)_i = σ²_p/n`
  - `optimize_cvar(returns, confidence=0.95)` → `QuantumPortfolioResult` — Scenario-based CVaR minimization via SLSQP
  - `_calculate_quantum_advantage()` → `float` — Heuristic speedup score
  - `compare_classical_quantum(returns, cov)` → `Dict` — Side-by-side weight/value/speedup comparison

---

### 4.7 `quantum_ml.py` (~996 lines)

**Key Imports**: `numpy`, `qiskit`, `sklearn` (SVC, RandomForestClassifier, metrics)

**Classical Fallbacks**: `ClassicalSVM` (RBF SVC), `ClassicalRandomForest`

**Dataclass**: `QuantumModelResult(accuracy, precision, recall, f1_score, predictions, prediction_probabilities, num_qubits, training_time_ms, model_type, backend_name, shots)`

**Classes**:

#### `QuantumSupportVectorMachine`
- **`__init__(self, num_qubits=4, shots=1024)`**
- Uses `ZZFeatureMap` → `FidelityQuantumKernel` → `QSVC`
- **Public Methods**: `fit(X, y)`, `predict(X)`, `decision_function(X)`

#### `VariationalQuantumClassifier`
- **`__init__(self, num_qubits=4, shots=1024)`**
- Uses `ZZFeatureMap` + `RealAmplitudes` ansatz, SPSA optimizer
- **Public Methods**: `fit(X, y)`, `predict(X)`, `predict_proba(X)`

#### `QuantumFeatureMap`
- **`__init__(self, num_features, encoding_type='amplitude'|'angle'|'pauli', reps=1)`**
- **Public Methods**: `create_feature_map()` → `QuantumCircuit`, `encode_data(x)` → `QuantumCircuit`

#### `QuantumMLPipeline`
- **`__init__(self, model_type='qsvm'|'vqc')`**
- **Public Methods**:
  - `prepare_data(X, y, test_size=0.2)` — Train/test split with scaling
  - `train()` → `QuantumModelResult`
  - `evaluate()` → `Dict` — accuracy, precision, recall, F1
  - `get_classification_report()` → `str`

**Convenience**: `create_financial_features()`, `quantum_market_prediction()`

---

## 5. data/ + data/live/ (10 files)

### 5.1 `__init__.py` (~72 lines)

**Exports**: `OHLCV`, `MarketDataLoader`, `DatabaseManager`, `DUCKDB_AVAILABLE`, all indicator functions (sma, ema, wma, dema, tema, rsi, macd, stochastic, williams_r, momentum, roc, atr, bollinger_bands, keltner_channels, adx, supertrend, obv, vwap, mfi)

---

### 5.2 `database.py` (~666 lines)

**Key Imports**: `duckdb`, `numpy`, `threading`

**Classes**:

#### `ConnectionPool`
- **`__init__(self, db_path, read_only=False, pool_size=5)`**
- Thread-safe connection pool with `checkout(timeout=30s)` / `checkin()`

#### `DatabaseManager`
- **`__init__(self, db_path=":memory:", read_only=False, pool_size=5)`**
- **Schema**: `ohlcv` (PK: symbol+timestamp), `trades` (PK: symbol+timestamp+id), `orderbook`, `backtests` (PK: id), `trade_log` (FK → backtests)
- **Public Methods**:
  - `insert_ohlcv(symbol, timestamps, O, H, L, C, V)` — INSERT OR REPLACE
  - `get_ohlcv(symbol, start_date, end_date)` → `Dict` — With optional date filtering
  - `get_symbols()` → `List[str]`
  - `calculate_returns(symbol, period=1)` → `ndarray` — SQL window `LAG` function
  - `calculate_volatility(symbol, window=20)` → `ndarray` — SQL `STDDEV` rolling
  - `correlation_matrix(symbols)` → `ndarray` — Cross-asset return correlations
  - `get_summary_stats(symbol)` → `Dict` — Annual return, vol, avg volume
  - `save_backtest(name, strategy, params, results)` → `str` — UUID-based
  - `get_backtest(backtest_id)` → `Dict`
  - `list_backtests()` → `List[Dict]`
  - `execute(query, params)`, `execute_df(query, params)` — Raw SQL
  - `vacuum()`, `close()`
- Context manager (`__enter__`/`__exit__`) support

---

### 5.3 `database_layer.py` (~190 lines)

**Key Imports**: `duckdb`, `dataclasses`

**Purpose**: Addresses Missing Concept 7.4 (OLAP/time-series DB)

**Dataclass**: `TimeSeriesRecord(timestamp, symbol, data: Dict)`

**Classes**:

#### `TimeSeriesDB`
- **`__init__(self, db_path='giga_system.duckdb')`**
- **Tables**: `ohlcv`, `signals`, `trades`, `portfolio_snapshots`
- **Public Methods**:
  - `insert_ohlcv(symbol, timestamp, O, H, L, C, V)` — Single-row insert
  - `insert_signal(timestamp, symbol, signal_name, value, confidence)`
  - `insert_trade(timestamp, symbol, side, quantity, price, strategy)`
  - `query_ohlcv(symbol, start, end)` → `List[Dict]`
  - `query_performance(strategy, period_days=30)` → `Dict` — Aggregated P&L stats
  - `export_to_parquet(table, output_path)` — `COPY ... TO 'path' (FORMAT PARQUET)`
  - `close()`
- Falls back to in-memory dict store if DuckDB unavailable

---

### 5.4 `indicators.py` (~775 lines)

**Key Imports**: `numpy`, `numba` (`@jit`)

All functions are standalone, JIT-compiled where possible.

**Moving Averages**:
- `sma(data, period)` — Simple moving average (rolling window)
- `ema(data, period)` — Exponential MA: `α = 2/(period+1)`
- `wma(data, period)` — Weighted MA (linear weights)
- `dema(data, period)` — Double EMA: `2·EMA - EMA(EMA)`
- `tema(data, period)` — Triple EMA: `3·EMA - 3·EMA² + EMA³`

**Momentum**:
- `rsi(data, period=14)` — Wilder smoothing, levels: 70 (overbought) / 30 (oversold)
- `macd(data, fast=12, slow=26, signal=9)` → `Tuple[ndarray, ndarray, ndarray]` — MACD line, signal, histogram
- `stochastic(high, low, close, k_period=14, d_period=3)` — %K/%D, levels: 80/20
- `williams_r(high, low, close, period=14)` — Range: 0 to -100
- `momentum(data, period=10)` — Simple rate of change
- `roc(data, period=10)` — Percentage rate of change

**Volatility**:
- `atr(high, low, close, period=14)` — `TR = max(H-L, |H-C_prev|, |L-C_prev|)`, Wilder smoothing
- `bollinger_bands(data, period=20, std_dev=2.0)` → `Tuple[upper, middle, lower]`
- `keltner_channels(high, low, close, period=20, multiplier=1.5)` — EMA ± ATR×multiplier

**Trend**:
- `adx(high, low, close, period=14)` — ADX + DI+/DI-, Levels: <20 weak, 20-40 developing, 40-60 strong, >60 very strong
- `supertrend(high, low, close, period=10, multiplier=3.0)` → `Tuple[ndarray, ndarray]` — Dynamic support/resistance with direction signal

**Volume**:
- `obv(close, volume)` — On-Balance Volume
- `vwap(high, low, close, volume)` — `cumulative(TP·V) / cumulative(V)`
- `mfi(high, low, close, volume, period=14)` — Volume-weighted RSI, levels: 80/20

---

### 5.5 `market_data.py` (~625 lines)

**Key Imports**: `numpy`, `yfinance`, `ccxt`, `polars`, `pandas`, `hashlib`

**Dataclass**: `OHLCV(symbol, timestamps, open, high, low, close, volume)`
- **Properties**: `returns`, `log_returns`, `typical_price`, `vwap`, `dollar_volume`, `intraday_range`, `overnight_return`, `intraday_return`

**Classes**:

#### `MarketDataLoader`
- **`__init__(self, cache_dir='./cache', api_keys=None)`**
- **Public Methods**:
  - `load(symbol, start, end, interval='1d', source=None)` → `OHLCV` — Auto-detect source
  - `_load_yahoo(symbol, start, end, interval)` — Via yfinance
  - `_load_alpha_vantage(symbol, start, end)` — REST API
  - `_load_local(path)` — CSV/Parquet files
  - `_load_crypto(symbol, start, end, interval, exchange='binance')` — Via ccxt with pagination + symbol normalization (handles `BTC/USDT`, `BTCUSDT`, `btcusdt`)
  - `_detect_source(symbol)` — Local file → crypto patterns → yahoo fallback
  - `_cache_key(symbol, start, end, interval, source)` → MD5 hash
  - `_load_from_cache(key)` — 1-day TTL, npz+json format
  - `_save_to_cache(key, data)`
  - `_validate(data)` — NaN checks + OHLC consistency (H≥L, H≥O, etc.)
  - `resample(data, freq='W'|'M'|'Q')` — Polars `group_by_dynamic` (preferred) or Pandas
  - `align(*datasets)` → `List[OHLCV]` — Common timestamp alignment

---

### 5.6 `multi_exchange.py` (~170 lines)

**Key Imports**: `ccxt`, `dataclasses`

**Purpose**: Addresses Missing Concept 7.5 (Binance+yfinance only)

**Dataclasses**: `ExchangePrice(exchange, bid, ask, last, volume, timestamp)`, `TriangulatedPrice(consensus_price, spread_bps, arbitrage_opportunity, best_bid_exchange, best_ask_exchange, prices)`

**Classes**:

#### `MultiExchangeData`
- **`__init__(self)`**
- **`SUPPORTED_EXCHANGES`**: 8 exchanges — binance, coinbasepro, kraken, bitfinex, okx, bybit, kucoin, gateio
- **Public Methods**:
  - `fetch_ticker(exchange_name, symbol)` → `ExchangePrice` — With synthetic fallback if exchange unavailable
  - `fetch_all_exchanges(symbol)` → `List[ExchangePrice]`
  - `triangulate_price(symbol)` → `TriangulatedPrice` — Volume-weighted consensus price + cross-exchange arbitrage detection (best_bid > best_ask)
  - `fetch_ohlcv(exchange_name, symbol, timeframe, limit)` → `List`

---

### 5.7 `preprocessing.py` (~652 lines)

**Key Imports**: `polars`, `numpy`

**Classes**:

#### `DataPreprocessor`
- **`__init__(self, date_column='timestamp', price_columns=['open','high','low','close'], volume_column='volume')`**
- **Public Methods**:
  - `clean_market_data(df, **kwargs)` → `pl.DataFrame` — Pipeline: datetime parse → sort → dedup → validate → handle missing → handle outliers → final validate
  - `calculate_returns(df, column='close', method='simple'|'log', periods=1)` → `pl.DataFrame`
  - `calculate_volatility(df, column='close', method='rolling_std'|'ewm'|'parkinson', window=20)` → `pl.DataFrame` — Parkinson: `σ_P = √(Σln(H/L)²/(4n·ln2))`
  - `create_features(df, feature_set='basic'|'technical'|'advanced')` → `pl.DataFrame`
    - basic: momentum, SMA ratios, volume ratio
    - technical: RSI, Bollinger Bands, MACD
    - advanced: Hurst exponent, fractal dimension, Shannon entropy
  - `resample_data(df, freq)` — `group_by_dynamic`
  - `_validate_price_data(df)` — Fix H<L by swapping
  - `_handle_missing_values(df, method='drop'|'forward_fill'|'interpolate')`
  - `_handle_outliers(df, method='iqr'|'zscore', threshold=3.0)` — Capping/winsorizing

**Helper Methods** (private):
- `_calculate_hurst(prices)` → `float` — R/S analysis, log-log regression
- `_calculate_fractal_dimension(prices)` → `float` — Higuchi method
- `_calculate_entropy(prices)` → `float` — Shannon entropy of discretized returns

**Convenience Functions**: `preprocess_market_data(df, config)`, `calculate_all_returns(df, price_column)`

---

### 5.8 `storage_manager.py` (~708 lines)

**Key Imports**: `duckdb`, `polars`, `warnings`, `re`, `pathlib`

**Constants**: `_ALLOWED_TABLES` (whitelist set), `_IDENTIFIER_RE` (regex for SQL injection protection)

**Classes**:

#### `StorageManager`
- **`__init__(self, db_path=None, memory_limit='1GB', threads=4)`**
- Default path: `giga_system.duckdb`
- **Schema (7 tables)**:
  - `market_data` — PK: timestamp+symbol
  - `options_chain` — With pre-computed Greeks (delta, gamma, theta, vega, rho), PK: timestamp+underlying+strike+expiry+option_type
  - `positions` — With Greek exposures (delta/gamma/theta/vega/rho_exposure)
  - `trades` — With execution_latency_ms
  - `performance_metrics` — With Greek P&L attribution (delta_pnl, gamma_pnl, theta_pnl, vega_pnl, rho_pnl)
  - `volatility_forecasts` — With model_parameters (JSON)
  - `audit_log` — Timestamped event logging
- **Public Methods**:
  - `execute_sql(sql, params)` — Parameterized execution
  - `query_to_polars(sql, params)` → `pl.DataFrame`
  - `insert_polars(table, df, mode='append'|'replace'|'ignore')` — Validated table name
  - `get_market_data(symbol, start, end)` → `pl.DataFrame` — Parameterized
  - `get_options_chain(underlying, expiry)` → `pl.DataFrame`
  - `get_portfolio_summary(user_id=1)` → `Dict` — Net Greeks + gross exposure
  - `record_trade(symbol, side, quantity, price, position_type, ...)` → `bool`
  - `get_performance_metrics(strategy_name, start, end)` → `pl.DataFrame`
  - `cleanup_old_data(days_to_keep=365)` — Purge old records
  - `get_database_info()` → `Dict` — Table sizes, file size, config
  - `close()`

**Global Instance**: `storage_manager = StorageManager()`, access via `get_storage()` / `initialize_storage(db_path)`

---

### 5.9 `data/live/binance_ws_feed.py` (~155 lines)

**Key Imports**: `websockets`, `ssl`, `json`, `asyncio`

**Classes**:

#### `BinanceWebSocketFeed`
- **`__init__(self, symbol='btcusdt')`**
- **Constants**: `MAX_RECONNECT_ATTEMPTS=20`, `INITIAL_BACKOFF_SEC=1.0`, `MAX_BACKOFF_SEC=60.0`
- **Stream**: `wss://stream.binance.com:9443/ws/<symbol>@aggTrade`
- **Public Methods**:
  - `connect()` — Async WebSocket with SSL verification, exponential backoff reconnection, ping/pong keepalive
  - `_process_msg(raw)` → `Dict` — Extracts: price, quantity, latency (event_time vs local), is_buyer_maker
  - `get_latest()` — Async queue consumer
  - `get_snapshot()` — Synchronous wrapper

---

### 5.10 `data/live/market_stream.py` (~130 lines)

**Key Imports**: `requests`, `threading`, `time`

**Classes**:

#### `TokenBucketRateLimiter`
- **`__init__(self, rate=5.0, burst=10)`**
- **Public Methods**: `acquire()` — Blocks until token available (token bucket algorithm)

#### `MarketStream`
- **`__init__(self, symbol='BTCUSDT', rate_limit=5.0)`**
- Uses Binance REST `GET /api/v3/ticker/price`
- **Public Methods**:
  - `start()` — Launches background thread
  - `stop()` — Graceful shutdown
  - `get_latest_tick()` → `Dict` — Latest price data
- **Error Handling**: Exponential backoff on errors, HTTP 429 detection, rate limiting via TokenBucket

---

## 6. bridge/ (7 files)

### 6.1 `__init__.py` (~80 lines)

**Exports**: `RSession`, `get_r_session`, `R_AVAILABLE`, `fit_arima`, `forecast_arima`, `test_stationarity`, `fit_garch`, `fit_egarch`, `fit_copula`, `fit_gpd`, `test_cointegration`, `granger_causality`, `fit_var`, `mean_variance_optimize`, `risk_parity`, `black_litterman`, `fit_hmm`, `markov_regime_switching`, `calculate_performance_metrics`, `drawdown_analysis`, `fit_dcc`, `tail_dependence`, `MarketData`, `DataBridge`, `StreamingDataSource`, `SimulatedTickStream`

---

### 6.2 `r_bridge.py` (~682 lines)

**Key Imports**: `numpy`, `rpy2.robjects`, `rpy2.robjects.packages`

**Classes**:

#### `RSession`
- **`__init__(self, r_scripts_path=None)`**
- **R Packages Loaded**: `forecast`, `rugarch`, `copula`, `evd`, `PortfolioAnalytics`, `PerformanceAnalytics`, `vars`, `urca`, `depmixS4`, `rmgarch`
- **R Scripts Sourced**: `timeseries_models.R`, `risk_modeling.R`, `econometrics.R`, `portfolio_optimization.R`, `performance_analytics.R`, `regime_detection.R`, `correlation_analysis.R`
- **Public Methods**:
  - `call(function_name, **kwargs)` → `Any` — Calls R function with automatic type conversion
  - `_convert_r_to_python(result)` — `ListVector → dict`, `FloatVector → ndarray`

**Module-Level Functions** (all use `get_r_session()`):

- **Time Series**: `fit_arima(data)` — auto.arima, `forecast_arima(model, h)` — point + intervals, `test_stationarity(returns)` — ADF + PP + KPSS with consensus
- **Volatility**: `fit_garch(returns, model='sGARCH', dist='std')` — rugarch ugarchspec/ugarchfit, `fit_egarch(returns)` — Asymmetric vol
- **Risk**: `fit_copula(x, y, family)` — gaussian/t/clayton/gumbel/frank, `fit_gpd(data, threshold)` — EVT tail modeling
- **Econometrics**: `test_cointegration(y, x)` — Engle-Granger, `granger_causality(x, y, max_lag)`, `fit_var(data, max_lag)` — VAR model
- **Portfolio**: `mean_variance_optimize(returns)` — Markowitz, `risk_parity(returns)`, `black_litterman(returns, views_P, views_Q, confidence)`
- **Regime**: `fit_hmm(returns, n_states=2)` — Hidden Markov, `markov_regime_switching(returns)` — Bull/bear
- **Performance**: `calculate_performance_metrics(returns, risk_free=0.02)`, `drawdown_analysis(returns, top_n=5)`
- **Correlation**: `fit_dcc(returns)` — DCC-GARCH, `tail_dependence(x, y, quantile=0.05)`

---

### 6.3 `model_wrapper.py` (~635 lines)

**Key Imports**: `numpy`, `polars`, `rpy2`, `arch` (Python GARCH), `statsmodels`

**⚠️ WARNING**: PHASE 2 — DO NOT IMPORT IN EXECUTION ENGINE (research only)

**Dataclass**: `ModelResults(success, parameters, fitted_values, residuals, log_likelihood, aic, bic, forecasts, forecast_errors)`

**Classes**:

#### `GARCHModel`
- **`__init__(self, model_type='sGARCH'|'eGARCH'|'gjrGARCH'|'apARCH', distribution='norm'|'std'|'sstd'|'ged')`**
- **Public Methods**:
  - `fit(returns)` → `ModelResults` — Via rugarch (R), falls back to `arch` Python package
  - `_fit_python_garch(returns)` → `ModelResults` — `arch_model(returns, vol='GARCH', p=1, q=1, dist=...)`
  - `forecast(model, horizon=10)` → `Dict` — Forward volatility forecasts with confidence intervals
  - `_extract_garch_results(r_model)` → `ModelResults`

#### `ARIMAModel`
- **`__init__(self, seasonal=False)`**
- **Public Methods**:
  - `fit(data)` → `ModelResults` — Via `auto.arima` (R), falls back to `statsmodels.ARIMA`
  - `_fit_python_arima(data)` → `ModelResults` — Manual order selection
  - `forecast(model, horizon=10)` → `Dict` — Point forecasts + confidence intervals
  - `_extract_arima_results(r_model)` → `ModelResults`

#### `CointegrationTest`
- **`__init__(self)`**
- Requires R package `urca`
- **Public Methods**:
  - `johansen_test(data, test_type='trace', k=2)` → `Dict` — `ca.jo()` from urca, returns test_statistic, critical_values, conclusion
  - `engle_granger_test(y, x)` → `Dict` — `ca.po()` from urca, two-series test
  - `_interpret_johansen(test_stat, crit_vals)` → `str` — Compare with 5% critical value

**Convenience Functions**: `fit_garch(returns)`, `fit_arima(data)`, `test_cointegration(data, method)`

---

### 6.4 `data_bridge.py` (~706 lines)

**Key Imports**: `numpy`, `polars`, `pandas`, `duckdb`, `json`, `pathlib`

**Dataclass**: `MarketData(symbol, timestamps, open, high, low, close, volume)`
- **Properties**: `returns` (simple), `log_returns`

**Classes**:

#### `DataBridge`
- **`__init__(self, data_dir='./data', cache_size=100)`**
- DuckDB backend with `market_data` table
- **Public Methods**:
  - `load_csv(path, symbol=None)` → `MarketData` — Polars (preferred) / Pandas fallback
  - `load_parquet(path, symbol=None)` → `MarketData`
  - `load_auto(path, symbol=None)` → `MarketData` — Auto-detect .csv/.parquet/.json
  - `_load_json(path, symbol)` → `MarketData`
  - `save_to_db(data: MarketData)` — DuckDB upsert (INSERT OR REPLACE)
  - `load_from_db(symbol, start, end)` → `MarketData` — Parameterized query
  - `query(sql)` → `pl.DataFrame` — Raw SQL access
  - `generate_synthetic(symbol='SYNTH', n_days=252, S0=100, mu=0.08, sigma=0.20)` → `MarketData` — GBM: `dS = μSdt + σSdW`
  - `validate(data)` → `Dict` — NaN checks, OHLC consistency, negative value checks
  - `get_cached(key)` / `cache(key, data)` / `clear_cache()` — LRU cache with size limit
  - `close()`

#### `StreamingDataSource` (ABC)
- **Abstract Methods**: `start()`, `stop()`, `ticks()` → `Iterator`

#### `SimulatedTickStream`
- **`__init__(self, symbol='SYNTH', tick_rate=10.0, initial_price=100.0, volatility=0.0001)`**
- Per-tick GBM: `price *= (1 + vol*Z)` at `tick_rate` Hz
- **Public Methods**: `start()`, `stop()`, `ticks()` → yields `{timestamp, symbol, price, volume}`

---

### 6.5 `data_converter.py` (~340 lines)

**Key Imports**: `numpy`, `polars`, `pandas`, `rpy2`

**Classes**:

#### `DataConverter`
- **Public Methods**:
  - `polars_to_r_dataframe(df, var_name)` — Polars → Pandas → R DataFrame, assigned to R variable
  - `r_dataframe_to_polars(var_name)` → `pl.DataFrame` — R → Pandas → Polars
  - `numpy_to_r_matrix(arr, var_name)` — NumPy → R matrix
  - `r_matrix_to_numpy(var_name)` → `ndarray`
  - `prepare_returns_for_garch(returns)` → `ndarray` — Clean NaN/infinite values, ensure float64
  - `get_r_model_results(model_var)` → `Dict` — Extract coef, fitted, residuals, logLik from R model
  - `_optimize_financial_types(df)` — Detect datetime/price/volume/return columns and cast appropriately
  - `_prepare_pandas_for_r(df)` — Clean column names (remove special chars)
  - `_validate_financial_array(arr)` → `bool` — Check finite + reasonable range

**Global Instance**: `data_converter = DataConverter()` + convenience functions

---

### 6.6 `research_live_bridge.py` (~480 lines)

**Key Imports**: `hashlib`, `time`, `json`, `pathlib`, `dataclasses`

**Purpose**: Addresses Missing Concepts 5.1–5.5 (research-to-live pipeline)

**Dataclasses**: `ResearchArtifact(run_id, timestamp, strategy_name, parameters, metrics, data_hash, validated, promoted_to_live)`, `PerformanceComparison(research_sharpe, live_sharpe, research_max_dd, live_max_dd, divergence, alert_level, recommendation)`

**Classes**:

#### `TOMLGenerator` (5.1)
- `generate_from_backtest(strategy_name, params, metrics)` → `str` — Creates `strategies_config.toml` with `[meta]`, `[execution_params]`, `[regime_params]`, `[strategy]` sections

#### `TOMLVersionManager` (5.2)
- `save_version(toml_str, label)` — SHA256 hash + timestamp file naming
- `list_versions()` → `List[Dict]`
- `rollback(version_file)` — Creates pre-rollback backup then restores

#### `TOMLParameterReader` (5.3)
- `load(path)` → `Dict` — Minimal TOML parser (no external dependency)
- `get(key, default)` — Dot-notation access: `"strategy.entry_threshold"`
- `verify_params_applied(config, live_params)` → `Dict` — Detect config-vs-live mismatches

#### `ResearchArtifactStore` (5.4)
- `save_artifact(artifact)` — JSON persistence
- `load_artifact(run_id)` → `ResearchArtifact`
- `list_artifacts()` → `List`
- `find_best(metric='sharpe_ratio', top_n=5)` → `List` — Rank by metric

#### `ResearchLiveComparator` (5.5)
- `compare(research_metrics, live_metrics, divergence_threshold=0.3)` → `PerformanceComparison`
- **Alert Levels**: 🔴 RED = stop trading, 🟡 YELLOW = reduce size, 🟢 GREEN = continue

---

### 6.7 `rpy2_interface.py` (~300 lines)

**Key Imports**: `rpy2.robjects`, `rpy2.robjects.packages`, `numpy`, `pandas`

**⚠️ WARNING**: PHASE 2 — AIR-GAP VIOLATION: Research code only, DO NOT DEPLOY TO LIVE EXECUTION.

**Classes**:

#### `RInterface`
- **Public Methods**:
  - `_initialize_r()` — Activate pandas2ri + numpy2ri converters
  - `_load_r_package(name)` — Import or install+import R package
  - `ensure_packages(packages)` → `bool` — Batch package installation
  - `execute_r_code(code)` → `Any` — Raw R code execution
  - `execute_r_script(path)` — `source()` an R script
  - `polars_to_r(df, var_name)` — Via Pandas intermediary
  - `r_to_polars(var_name)` → `pl.DataFrame`
  - `numpy_to_r(arr, var_name)` — Direct numpy→R
  - `r_to_numpy(var_name)` → `ndarray`
- **Required R Packages**: forecast, rugarch, vars, urca, PerformanceAnalytics, quantmod, TTR, MASS, copula, evd

**Global Instance**: `r_interface = RInterface()`

**Utilities**: `check_r_availability()`, `get_r_version()`, `install_required_r_packages()`

---

## 7. execution/ (8 files)

### 7.1 `__init__.py` (~25 lines)

**Exports**: `OrderManager`, `Order`, `OrderType`, `OrderSide`, `OrderStatus`, `ExecutionEngine`, `FillModel`, `FillResult`, `SmartOrderRouter`, `Venue`, `VenueMetrics`, `OrderRouter`, `LatencyMonitor`, `LatencyComponent`, `LatencyMeasurement`, `LatencyStats`

---

### 7.2 `execution_engine.py` (~280 lines)

**Key Imports**: `numpy`, `time`

**Enum**: `FillModel(IMMEDIATE, LINEAR, SQRT, REALISTIC, CHAOTIC)`

**Dataclass**: `FillResult(filled_quantity, avg_fill_price, total_slippage_bps, market_impact_bps, latency_us, timestamp, partial_fill=False, status='FILLED', error_message=None)`

**Classes**:

#### `ExecutionEngine`
- **`__init__(self, fill_model=FillModel.REALISTIC, base_latency_us=500, latency_std_us=100)`**
- **Phase 11**: Chaos mode activated when `fill_model == CHAOTIC`
- **Public Methods**:
  - `execute_order(quantity, target_price, market_volume=1_000_000, volatility=0.02, urgency=0.5)` → `FillResult` — Full execution simulation with impact + slippage + latency
  - `get_execution_stats()` → `Dict` — Total fills, avg slippage, avg latency

**Private Methods**:
  - `_apply_chaos(quantity, price)` → `Optional[FillResult]` — Phase 11: 2% random rejection, 1% ghost order/timeout
  - `_determine_fill_quantity(quantity, market_volume, urgency)` — Partial fills for large orders relative to volume. Chaos: 20% partial fill chance
  - `_calculate_market_impact(quantity, market_volume, volatility, urgency)` → `float` — **Almgren-Chriss**: `impact = σ·√(participation) · 10000 · urgency` + permanent component
  - `_calculate_timing_slippage(urgency, volatility)` → `float` — Price drift during execution: `drift_bps = σ·√(t/(252·6.5))·10000`
  - `_calculate_spread_cost(urgency, volatility)` → `float` — `spread_cost = σ·100·urgency`

**Chaos Features**: Latency spikes (10% chance, 5–50× multiplier), random rejections (2%), ghost orders (1%), forced partial fills (20%)

---

### 7.3 `instructions.py` (~88 lines)

**Key Imports**: `dataclasses`, `enum`, `datetime`

**Purpose**: Stage 4 "Finite Actions" — deterministic execution commands reduced from probabilistic signals.

**Enums**: `ActionType(BUY, SELL, CANCEL, HALT)`, `InstructionStatus(PENDING, SUBMITTED, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED, EXPIRED)`

**Dataclass**:

#### `ExecutionInstruction`
- **Fields**: `id`, `action: ActionType`, `asset`, `quantity: float`, `limit_price`, `stop_price`, `created_at`, `expires_at`, `status`, `source_artifact_id` (traceability), `filled_quantity`, `avg_fill_price`, `metadata`
- **Methods**:
  - `validate()` → `bool` — Well-formed checks (positive quantity, valid prices, expiry after creation)
  - `is_active` (property) — Not in terminal status and not expired
  - `fill_ratio` (property) — `filled_quantity / quantity`

---

### 7.4 `latency_monitor.py` (~320 lines)

**Key Imports**: `time`, `numpy`, `collections.deque`, `contextlib`

**Enum**: `LatencyComponent(DATA_FEED, SIGNAL_GENERATION, RISK_CHECK, ORDER_SUBMISSION, VENUE_ROUTING, FILL_PROCESSING, TOTAL_LOOP)`

**Dataclasses**: `LatencyMeasurement(component, latency_us, timestamp, metadata)`, `LatencyStats(component, count, mean_us, median_us, p95_us, p99_us, max_us, std_us)`

**Classes**:

#### `LatencyMonitor`
- **`__init__(self, window_size=10000, alert_threshold_us=None)`**
- **Default Alert Thresholds** (microseconds):
  - DATA_FEED: 1000 (1ms)
  - SIGNAL_GENERATION: 500
  - RISK_CHECK: 100
  - ORDER_SUBMISSION: 200
  - VENUE_ROUTING: 500
  - FILL_PROCESSING: 300
  - TOTAL_LOOP: 5000 (5ms)
- **Public Methods**:
  - `start_timer(timer_id)` — Begin perf_counter measurement
  - `stop_timer(timer_id, component, metadata)` → `float` — Record latency in μs
  - `track(component, metadata)` — **Context manager** for easy timing: `with monitor.track(LatencyComponent.RISK_CHECK): ...`
  - `record_latency(component, latency_us, metadata)` — Direct recording
  - `get_stats(component=None)` → `Dict[LatencyComponent, LatencyStats]` — Mean, median, P95, P99, max, std
  - `get_recent_alerts(n=10)` → `List[Dict]`
  - `get_alert_summary()` → `Dict` — Alerts by component
  - `detect_degradation(component, lookback_recent=100, lookback_baseline=1000)` → `Dict` — Compares recent vs baseline: degradation if mean increase >20% or P95 increase >30%
  - `reset()` — Clear all data

---

### 7.5 `order_manager.py` (~500 lines)

**Key Imports**: `time`, `numpy`, `threading`, `logging`

**Enums**: `OrderType(MARKET, LIMIT, STOP, STOP_LIMIT, IOC, FOK, ICEBERG)`, `OrderSide(BUY, SELL)`, `OrderStatus(PENDING, SUBMITTED, ACKNOWLEDGED, PARTIAL_FILL, FILLED, CANCELLED, REJECTED)`

**Dataclass**:

#### `Order`
- **Fields**: `symbol`, `side: OrderSide`, `order_type: OrderType`, `quantity`, `price`, `stop_price`, `order_id` (auto: `ORD_{timestamp_us}`), timestamps (created/submitted/acked/filled), `status`, `filled_quantity`, `avg_fill_price`, latencies (submit/ack/fill), `max_slippage_bps=10`, `time_in_force='DAY'`
- **Methods**: `is_active()`, `remaining_quantity()`, `calculate_latency(phase)` — Supports Phase 11 ACK timing

**Classes**:

#### `OrderManager`
- **`__init__(self, initial_capital=1_000_000)`**
- Thread-safe via `threading.Lock()`
- **Risk Limits**: `max_order_value=100,000`, `max_daily_loss=50,000`, `max_position_size` per symbol
- **Public Methods**:
  - `pre_trade_risk_check(order, current_price)` → `Tuple[bool, str]` — **Target: <100μs**. 5 checks: order value, daily loss, position limit, capital sufficiency, slippage tolerance
  - `submit_order(order, current_price)` → `Tuple[bool, str]` — Risk check + submit with latency tracking
  - `fill_order(order_id, fill_price, fill_quantity)` → `bool` — Position update with **weighted average price** (BUG#5 FIX). Proper P&L realization for both long closes and short covers (BUG#3 FIX). Handles position flips (long→short, short→long)
  - `cancel_order(order_id)` → `bool`
  - `cancel_all_orders()` → `int`
  - `acknowledge_order(order_id)` → `bool` — Phase 11: exchange ACK
  - `get_position(symbol)` → `int`
  - `get_pnl(current_prices)` → `float` — Unrealized + realized
  - `reset_daily_pnl()` — Start-of-day reset
  - `get_position_summary()` → `Dict`
  - `get_metrics()` → `Dict` — Fill rate, avg latency, capital, daily P&L

#### `ExposureGovernor`
- **`__init__(self, max_exposure_per_symbol=0.30)`**
- Phase 10 safety logic
- `check_exposure(symbol, proposed_value, current_equity, current_positions)` → `bool` — Returns `True` if safe

---

### 7.6 `order_router.py` (~85 lines)

**Key Imports**: `logging`, `time`

**Classes**:

#### `OrderRouter`
- **`__init__(self, executor, default_symbol='BTCUSDT')`**
- Phase 6: Execution Gateway — routes signals to exchange executor
- **Public Methods**:
  - `route_order(signal, current_price=0.0, quantity_override=0.0)` → `Dict` — Converts brain signal to exchange order. Supports `symbol` from signal dict (multi-symbol). Determines side from signal action: `"ENTRY"→BUY`, else `SELL`
  - `route_batch(signals, current_prices)` → `List[Dict]` — Batch routing
  - `stats` (property) → `Dict` — Orders routed, active routes

---

### 7.7 `smart_router.py` (~340 lines)

**Key Imports**: `numpy`, `logging`

**Enum**: `Venue(BINANCE, COINBASE, KRAKEN, OKX, BYBIT, KUCOIN)`

**Classes**:

#### `SlippageModel`
- **Static Method**: `calculate_effective_price(mid_price, side, quantity, volatility, spread, daily_volume)` → `float`
- **Formula**: `Price = Mid ± (Spread/2 + Impact)`, `Impact = σ·√(qty/(10%·daily_vol))·mid_price`

#### `VenueMetrics`
- **Fields**: `venue`, `avg_latency_ms`, `liquidity_score`, `maker_fee_bps`, `taker_fee_bps`, `fill_rate`, `avg_slippage_bps`
- `calculate_score(order_size_usd, urgency)` → `float` — Weighted score: `0.35·latency + 0.30·cost + 0.20·liquidity + 0.10·fill_rate + 0.05·slippage` (urgency shifts weight between latency and cost)

#### `SmartOrderRouter`
- **`__init__(self)`** — Pre-loaded with 6 venue metrics:
  - Binance: 50ms latency, 0.98 liquidity, 1.0/1.0 bps fees
  - Coinbase: 80ms, 0.90, 4.0/6.0 bps
  - Kraken: 100ms, 0.85, 1.6/2.6 bps
  - OKX: 60ms, 0.93, 0.8/1.0 bps
  - Bybit: 55ms, 0.91, 1.0/1.0 bps
  - KuCoin: 90ms, 0.80, 1.0/1.0 bps
- **Public Methods**:
  - `route_order(symbol, quantity, urgency, max_venues=3, price)` → `List[Tuple[Venue, qty, fee]]` — Score-based allocation across top venues
  - `get_best_venue(symbol, quantity, urgency)` → `Venue` — Single best venue
  - `update_venue_metrics(venue, **kwargs)` — EMA update: `new = 0.9·old + 0.1·observed`
  - `get_routing_stats()` → `Dict` — Volume and order distribution across venues

#### `SlicingEngine`
- Phase 10: Order Slicing by regime
- `slice_order(total_qty, regime)` → `List[float]` — SEED/GROWTH: single fill. SCALE+: TWAP-style chunks (max 0.5 BTC per slice)

---

### 7.8 `binance_executor.py` (~500 lines)

**Key Imports**: `ccxt`, `hashlib`, `hmac`, `time`, `os`, `logging`

**Classes**:

#### `BinanceExecutor`
- **`__init__(self, api_key=None, api_secret=None, paper_mode=True, testnet=True)`**
- **Modes**:
  - **Paper**: Uses `ExecutionEngine` with `FillModel.CHAOTIC` for simulation
  - **Live**: Uses `ccxt.binance` with `sandbox_mode` for testnet
- **API Keys**: From env vars `BINANCE_API_KEY` / `BINANCE_API_SECRET` or constructor
- **Public Methods**:
  - `_init_ccxt_exchange()` — ccxt configuration with rate limiting, headers
  - `sign_request(params)` → `str` — HMAC SHA256 signature
  - `post_order(symbol, side, quantity, price=0)` → `Dict` — Market/limit. Paper: internal tracking. Live: `ccxt.create_order()`
  - `_live_post_order(symbol, side, quantity, price)` → `Dict` — ccxt bridge
  - `cancel_order(order_id, symbol)` / `cancel_all_orders()` → `Dict`
  - `reconcile_positions(symbol)` → `Dict` — Sync local state vs exchange positions
  - `fetch_balance()` → `Dict` — Paper: internal, Live: `ccxt.fetch_balance()`
  - `_simulate_network_call()` — Phase 11: Random latency simulation + chaos events
  - `_normalise_symbol(raw)` → `str` — Converts `BTCUSDT`/`btcusdt` → `BTC/USDT` for ccxt

---

## 8. risk/ (2 files)

### 8.1 `session_guard.py` (~185 lines)

**Key Imports**: `time`, `logging`, `typing`

**Classes**:

#### `SessionGuard`
- **`__init__(self, max_loss=500.0, max_orders_per_min=5, max_drawdown_pct=0.05, max_session_hours=24.0)`**
- Phase 6: Survival Mechanism — Kill switch for live sessions
- **Wiring**: `wire(order_manager=None, exchange=None)` — Connects to live components for real action
- **Public Methods**:
  - `register_shutdown_callback(callback)` — Hook for alert systems, logging
  - `initialize_account(initial_equity)` — Set starting equity and peak
  - `check_health(current_equity)` → `bool` — Returns `False` if system must stop. 3 checks:
    1. **Absolute Drawdown**: `start_equity - current > max_loss`
    2. **Percentage Drawdown from Peak**: `(peak - current)/peak > max_drawdown_pct`
    3. **Session Time Limit**: `elapsed_hours > max_session_hours`
  - `validate_order_rate()` → `bool` — Rolling 60s window rate limiter
  - `emergency_shutdown()` — **REAL KILL SWITCH**: Step 1: Cancel all active orders via `order_manager.cancel_order()`. Step 2: Flatten all positions via `exchange.post_order()` (market sells/buys). Step 3: Fire all registered shutdown callbacks
  - `get_drawdown_details()` → `Dict` — Current/start/peak equity, absolute/pct DD
  - `get_status()` → `Dict` — `"KILLED"` or `"ACTIVE"` with reason
  - `reset(new_equity=None)` — Manual review acknowledged reset

---

### 8.2 `strategy_breaker.py` (~185 lines)

**Key Imports**: `logging`, `time`

**Purpose**: Per-strategy circuit breaker — isolates strategy-level failures so one broken strategy doesn't take down the portfolio.

**Classes**:

#### `StrategyBreaker`
- **`__init__(self, name, max_consecutive_losses=5, max_drawdown_pct=0.10, daily_loss_limit=200.0, cooldown_seconds=300.0)`**
- **Public Methods**:
  - `allow_trade()` → `bool` — `False` if tripped (auto-resets after cooldown)
  - `record_result(pnl)` — Check 3 trip conditions:
    1. **Consecutive Losses**: `≥ max_consecutive_losses`
    2. **Drawdown from Peak**: `(peak_pnl - current_pnl)/peak_pnl ≥ max_drawdown_pct`
    3. **Daily Loss Cap**: `daily_pnl ≤ -daily_loss_limit`
  - `reset()` — Manual CLOSED reset
  - `state` (property) → `"OPEN"` or `"CLOSED"`
  - `info` (property) → `Dict` — Full state details

#### `StrategyBreakerManager`
- **`__init__(self, max_consecutive_losses=5, max_strategy_drawdown_pct=0.10, daily_loss_limit=200.0, cooldown_seconds=300.0)`**
- Auto-registers strategies on first access
- **Public Methods**:
  - `allow_trade(strategy_name)` → `bool`
  - `record_result(strategy_name, pnl)`
  - `get_all_states()` → `Dict[str, Dict]`
  - `reset(strategy_name)` / `reset_all()`

---

## 9. account/ (1 file)

### 9.1 `live_account.py` (~240 lines)

**Key Imports**: `time`, `logging`

**Classes**:

#### `LiveAccount`
- **`__init__(self, start_balance=10000.0, fee_rate=0.0004, max_leverage=1.0, maintenance_margin_rate=0.005)`**
- **Fee Rate**: 0.04% (taker fee), deducted on every trade
- **Public Methods**:
  - `update_mark_price(symbol, price)` → `float` — Update unrealized P&L for all positions. Linear P&L: `(mark - entry) × size`. Returns updated equity
  - `execute_trade(symbol, side, price, size)` → `Tuple[realized_pnl, fee]` —
    - Fee: `notional × fee_rate`, deducted from cash
    - **Position Management**:
      - Increasing position → weighted average entry price
      - Closing position → realized P&L (long: `(price-entry)×qty`, short: `(entry-price)×qty`)
      - Position flip → closes existing + opens new at current price
    - Records full trade history
  - `get_equity()` → `float` — `cash + unrealized_pnl`
  - `get_total_pnl()` → `float` — `total_realized_pnl - total_fees`
  - `get_position_summary()` → `Dict` — All open positions with side/notional
  - `get_account_summary()` → `Dict` — Full summary: cash, equity, P&L, fees, return%, leverage, margin metrics
  - `get_gross_exposure()` → `float` — Total notional across all positions
  - `get_margin_info()` → `Dict` — `current_leverage`, `margin_used`, `margin_available`, `margin_ratio`, `can_open_more` (ratio < 0.95)
  - `check_leverage_limit(symbol, additional_notional)` → `bool` — Projected leverage check
  - `estimate_liquidation_price(symbol)` → `Optional[float]` —
    - Long: `entry × (1 - 1/leverage + mmr)`
    - Short: `entry × (1 + 1/leverage - mmr)`

---

## 10. brain/ (1 file)

### 10.1 `state_machine.py` (~225 lines)

**Key Imports**: `enum`, `time`, `logging`

**Enums**: `State(BOOT, IDLE, ANALYZING, ENTRY_SIGNAL, IN_POSITION, EXIT_SIGNAL, COOLDOWN, HALTED)`, `MarketRegime(LOW_VOL, NORMAL_VOL, HIGH_VOL, CRASH)`

**Classes**:

#### `StateMachineBrain`
- **`__init__(self, config: Dict)`**
- **Config Keys**: `max_holding_bars` (default 200), `trailing_stop_pct` (default 0.02/2%), `stop_loss_pct` (default 0.03/3%), `take_profit_pct` (default 0.06/6%), `cooldown_seconds` (default 5.0)
- **State Machine**: `BOOT → IDLE → ANALYZING → ENTRY_SIGNAL → IN_POSITION → EXIT_SIGNAL → COOLDOWN → IDLE` (cycle). `HALTED` on CRASH regime (manual or IV-drop reset).
- **Public Methods**:
  - `update(market_data)` → `Optional[Dict]` — Main brain tick. Returns action dict or None.
    - **State Transitions**:
      - `BOOT → IDLE` (immediate)
      - `IDLE → ANALYZING` (normal) or `IDLE → HALTED` (crash)
      - `ANALYZING → ENTRY_SIGNAL` (if signal generated)
      - `ENTRY_SIGNAL → IN_POSITION` (returns `{"action": "EXECUTE_ENTRY", ...}`)
      - `IN_POSITION → EXIT_SIGNAL` (if exit condition met, returns `{"action": "EXECUTE_EXIT", ...}`)
      - `EXIT_SIGNAL → COOLDOWN` (cooldown timer set)
      - `COOLDOWN → IDLE` (after cooldown expires)
      - `HALTED → IDLE` (when IV < 1.0)
  - `get_state_summary()` → `Dict` — Full state snapshot for monitoring
  - `reset_position()` — Clear all position-related state

**Private Methods**:
  - `_detect_regime(iv)` — IV thresholds: `>2.0=CRASH`, `>1.0=HIGH_VOL`, `<0.3=LOW_VOL`, else `NORMAL_VOL`
  - `_evaluate_entry(tick)` → `Optional[str]` — Phase 8 AI hook: checks `ai_decision` first, then fallback regime-based logic:
    - LOW_VOL + delta>0.6 → `LONG_CALL_MOMENTUM`
    - NORMAL_VOL + delta>0.52 → `LONG_CALL_TREND`
    - NORMAL_VOL + delta<0.48 → `SHORT_CALL_TREND`
  - `_evaluate_exit(tick)` → `Optional[str]` — 5 exit conditions:
    1. **Stop-Loss**: Price crosses stop (with trailing for both long/short)
    2. **Take-Profit**: Price reaches target
    3. **Time-Based**: `bars_in_position ≥ max_bars_in_position`
    4. **Regime Change**: CRASH detected
    5. **AI-Driven**: `ai_decision` with EXIT action

**Trailing Stop Logic**: Long: ratchets up `high_water_mark × (1 - trailing_pct)`. Short: ratchets down `low_water_mark × (1 + trailing_pct)`.

---

## Summary Statistics

| Directory | Files | Est. Lines | Key Components |
|---|---|---|---|
| research/core/ | 22 | ~10,500 | BS pricing, Greeks (JIT), Monte Carlo, Vol surface, Alpha signals, Stochastic models, Information geometry |
| research/ml/ | 4 | ~1,535 | Feature engineering (30+), Regime detection (GMM/HMM), Vol forecasting (EWMA/GARCH/HAR) |
| research/strategies/ | 7 | ~2,925 | Market making (Avellaneda-Stoikov), Momentum, Options (delta hedge, vol arb, iron condor), Pairs trading |
| research/quantum/ | 7 | ~4,551 | Quantum Monte Carlo, QAOA/VQE optimization, Quantum risk, Hybrid algorithms, QML (QSVM/VQC) |
| data/ + data/live/ | 10 | ~4,145 | DuckDB storage, JIT indicators (20+), Multi-exchange, Polars preprocessing, WebSocket feed |
| bridge/ | 7 | ~3,223 | Python↔R bridge (rpy2), GARCH/ARIMA wrappers, Research→Live pipeline (TOML), Data conversion |
| execution/ | 8 | ~2,138 | Fill simulation (Almgren-Chriss), Order manager (<100μs risk checks), Smart routing (6 venues), Latency monitor |
| risk/ | 2 | ~370 | Session kill switch (DD/rate/time limits), Per-strategy circuit breaker |
| account/ | 1 | ~240 | Live P&L, margin/leverage tracking, liquidation estimation |
| brain/ | 1 | ~225 | FSM (8 states), Regime detection, Entry/exit logic with trailing stops |
| **TOTAL** | **69** | **~29,852** | |

---

## Key Architectural Patterns

1. **JIT Compilation**: Core pricing/Greeks/indicators use Numba `@jit(nopython=True, cache=True)` for microsecond-level performance
2. **Quantum-Classical Hybrid**: All quantum algorithms have classical fallbacks — system works without Qiskit
3. **Python-R Bridge**: Statistical modeling via rpy2 with Python fallbacks (arch, statsmodels)
4. **Multi-Phase Architecture**: References to Phase 3 (lifecycle), Phase 6 (execution gateway), Phase 8 (AI hooks), Phase 10 (safety), Phase 11 (chaos testing)
5. **Polars-First**: Data processing prefers Polars over Pandas, with Pandas fallbacks
6. **DuckDB OLAP**: SQL-based analytics with parameterized queries and SQL injection protection
7. **BUG Fix Annotations**: Explicit `BUG#3`, `BUG#4`, `BUG#5` fix comments indicating resolved issues
8. **Research/Live Air Gap**: Explicit warnings preventing research code from leaking into execution engine
