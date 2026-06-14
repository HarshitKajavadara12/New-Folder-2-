# Greek Mathematics in Finance

## A Historical Journey from Ancient Greece to Wall Street

> "Let no one ignorant of geometry enter here." — Inscription at Plato's Academy

---

## Introduction

The ancient Greeks developed mathematical concepts that form the foundation of modern quantitative finance. This guide explores how 2,500-year-old ideas power today's trillion-dollar derivatives markets.

---

## The Greek Letters (Greeks)

In options trading, we use Greek letters to represent sensitivities of option prices to various factors. This naming convention honors the mathematical heritage.

### Delta (Δ) - First Derivative

**Historical Context:**
Archimedes (287-212 BCE) developed methods for finding tangent lines to curves, essentially computing derivatives geometrically.

**Financial Meaning:**
Delta measures the rate of change of option value with respect to the underlying asset price.

```
Δ = ∂V/∂S

For a call option:
Δ_call = N(d₁) ∈ [0, 1]

For a put option:
Δ_put = N(d₁) - 1 ∈ [-1, 0]
```

**Interpretation:**
- Delta ≈ 0.5 → Option is at-the-money
- Delta ≈ 1.0 → Deep in-the-money call (moves like stock)
- Delta ≈ 0.0 → Deep out-of-the-money (unlikely to exercise)

**Greek Connection:**
Just as Archimedes found the slope of a tangent to a parabola, Delta gives us the "slope" of the option price curve.

```python
# Delta calculation
def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)
```

---

### Gamma (Γ) - Second Derivative

**Historical Context:**
Apollonius of Perga (262-190 BCE) studied conic sections and their curvature, laying groundwork for second-order analysis.

**Financial Meaning:**
Gamma measures the rate of change of Delta with respect to the underlying price.

```
Γ = ∂²V/∂S² = ∂Δ/∂S

Γ = N'(d₁) / (S × σ × √T)
```

**Interpretation:**
- High Gamma → Delta changes rapidly (dangerous near expiry)
- Gamma is always positive for long options
- Peak Gamma occurs at-the-money

**The Gamma-Convexity Connection:**
Gamma represents convexity. Positive Gamma means you benefit from large moves in either direction.

```
Long Gamma:   when markets move big
Short Gamma:   when markets move big
```

**Greek Connection:**
Apollonius' study of how curves bend relates directly to Gamma's measure of price convexity.

---

### Theta (Θ) - Time Decay

**Historical Context:**
Heraclitus (535-475 BCE) philosophized about time and change: "You cannot step into the same river twice."

**Financial Meaning:**
Theta measures the rate of change of option value with respect to time.

```
Θ = ∂V/∂t

Θ_call = -[S × N'(d₁) × σ / (2√T)] - r × K × e^(-rT) × N(d₂)
```

**Interpretation:**
- Theta is typically negative for long options (time decay)
- Accelerates as expiration approaches
- At-the-money options have highest Theta

**Time Value Erosion:**
```
Days to Expiry    Daily Theta
30               $0.05
7                $0.12
1                $0.50
```

**Greek Connection:**
Heraclitus understood that time changes everything. Theta quantifies exactly how much.

---

### Vega (ν) - Volatility Sensitivity

**Historical Context:**
While "Vega" isn't actually a Greek letter, we include it because the Greeks understood variability in nature.

**Financial Meaning:**
Vega measures sensitivity to implied volatility changes.

```
ν = ∂V/∂σ

ν = S × √T × N'(d₁)
```

**Interpretation:**
- Long options have positive Vega (benefit from vol increase)
- Vega is highest for at-the-money options
- Longer-dated options have higher Vega

```python
def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)
```

---

### Rho (ρ) - Interest Rate Sensitivity

**Historical Context:**
The concept of interest and growth was understood by Greeks through geometric series.

**Financial Meaning:**
Rho measures sensitivity to interest rate changes.

```
ρ = ∂V/∂r

ρ_call = K × T × e^(-rT) × N(d₂)
ρ_put = -K × T × e^(-rT) × N(-d₂)
```

**Interpretation:**
- Calls have positive Rho (benefit from rate increases)
- Puts have negative Rho
- Rho is more significant for long-dated options

---

## Second-Order Greeks

### Vanna - Cross-Derivative

```
Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S
```

**Meaning:** How Delta changes with volatility, or how Vega changes with spot.

### Volga (Vomma) - Vol of Vol

```
Volga = ∂²V/∂σ² = ∂ν/∂σ
```

**Meaning:** Convexity in volatility space. Important for vol surface dynamics.

### Charm - Delta Decay

```
Charm = ∂²V/∂S∂t = ∂Δ/∂t
```

**Meaning:** How Delta changes over time. Critical for weekend/holiday risk.

### Speed - Third Derivative

```
Speed = ∂³V/∂S³ = ∂Γ/∂S
```

**Meaning:** Rate of change of Gamma. Important for large moves.

---

## The Geometry of Black-Scholes

### The PDE as a Heat Equation

Black-Scholes is related to the heat equation, studied by Fourier but with roots in Greek physics:

```
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

Transform: Let V = e^(-rτ)u, S = Ke^x, τ = T - t

This becomes:
```
∂u/∂τ = (1/2)σ²(∂²u/∂x²)
```

The heat equation! Greeks understood diffusion of heat intuitively.

### Option Price as an Integral

The Black-Scholes formula is essentially:

```
V = e^(-rT) × E[max(S_T - K, 0)]

V = e^(-rT) × ∫_{-∞}^{∞} max(S_T - K, 0) × φ(z) dz
```

This is a weighted average (expectation) — a concept the Greeks pioneered.

---

## Greek Mathematical Concepts in Finance

### 1. The Golden Ratio (φ) in Markets

```
φ = (1 + √5) / 2 ≈ 1.618

Fibonacci levels: 23.6%, 38.2%, 61.8%, 78.6%
```

While controversial, Fibonacci retracements appear in technical analysis.

### 2. Euclidean Distance in Portfolio Theory

```
Distance = √(Σ(x_i - y_i)²)
```

Used in clustering assets, measuring portfolio similarity.

### 3. Pythagorean Theorem in Risk

```
Portfolio σ² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂
```

When ρ = 0: σ_p = √(σ₁² + σ₂²) — Pythagoras!

### 4. Geometric Series in Present Value

```
PV = Σ(C / (1+r)^t) = C × (1 - (1+r)^(-n)) / r
```

Archimedes summed infinite geometric series.

---

## The Greeks in Practice

### Delta Hedging

Maintain Δ = 0 to be direction-neutral:

```python
def delta_hedge(option_delta, shares_held, option_quantity):
    """Calculate shares needed to delta hedge."""
    target_shares = -option_delta * option_quantity
    trade = target_shares - shares_held
    return trade
```

### Gamma Scalping

Profit from realized volatility exceeding implied:

```
P&L = (1/2) × Γ × (ΔS)² - Θ × Δt

If (1/2) × Γ × (ΔS)² > Θ × Δt → Profit!
```

### Vega Trading

Trade implied vs. realized volatility:

```python
def vega_pnl(vega, iv_change):
    """P&L from volatility change."""
    return vega * iv_change * 100  # Per 1% IV change
```

---

## Summary: Greeks Cheat Sheet

| Greek | Symbol | Formula | Measures |
|-------|--------|---------|----------|
| Delta | Δ | ∂V/∂S | Price sensitivity |
| Gamma | Γ | ∂²V/∂S² | Delta sensitivity |
| Theta | Θ | ∂V/∂t | Time decay |
| Vega | ν | ∂V/∂σ | Volatility sensitivity |
| Rho | ρ | ∂V/∂r | Rate sensitivity |
| Vanna | | ∂²V/∂S∂σ | Delta-vol cross |
| Volga | | ∂²V/∂σ² | Vol convexity |
| Charm | | ∂²V/∂S∂t | Delta decay |

---

## Conclusion

From Euclid's geometry to modern derivatives, Greek mathematical thinking pervades quantitative finance. Understanding these connections isn't just historical curiosity — it provides deeper intuition for complex financial instruments.

> "The laws of nature are but the mathematical thoughts of God." — Euclid

In finance, the Greeks help us understand the mathematical structure of uncertainty itself.

---

*GIGA SYSTEM — Honoring 2,500 years of mathematical tradition*
