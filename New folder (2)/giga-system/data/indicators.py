"""
GIGA SYSTEM - Technical Indicators
High-performance indicator calculations using NumPy and Numba
"""

import numpy as np
from typing import Tuple, Optional
from functools import lru_cache

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f
    prange = range


# =============================================================================
# MOVING AVERAGES
# =============================================================================

@njit(cache=True)
def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average.
    
    SMA_t = (1/n) * Σ(P_{t-i}) for i = 0 to n-1
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    period : int
        Lookback period.
    
    Returns
    -------
    np.ndarray
        SMA values (NaN for first period-1 values).
    """
    n = len(prices)
    result = np.empty(n)
    result[:period-1] = np.nan
    
    # Initial sum
    window_sum = np.sum(prices[:period])
    result[period-1] = window_sum / period
    
    # Rolling calculation
    for i in range(period, n):
        window_sum = window_sum - prices[i-period] + prices[i]
        result[i] = window_sum / period
    
    return result


@njit(cache=True)
def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average.
    
    EMA_t = α * P_t + (1-α) * EMA_{t-1}
    where α = 2 / (period + 1)
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    period : int
        Lookback period.
    
    Returns
    -------
    np.ndarray
        EMA values.
    """
    n = len(prices)
    result = np.empty(n)
    alpha = 2.0 / (period + 1)
    
    # Initialize with SMA
    result[0] = prices[0]
    
    for i in range(1, n):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    
    return result


@njit(cache=True)
def wma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Weighted Moving Average.
    
    WMA_t = Σ(w_i * P_{t-i}) / Σ(w_i)
    where w_i = period - i (linear weights)
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    period : int
        Lookback period.
    
    Returns
    -------
    np.ndarray
        WMA values.
    """
    n = len(prices)
    result = np.empty(n)
    result[:period-1] = np.nan
    
    # Weights: [period, period-1, ..., 1]
    weights = np.arange(period, 0, -1, dtype=np.float64)
    weight_sum = np.sum(weights)
    
    for i in range(period-1, n):
        weighted_sum = 0.0
        for j in range(period):
            weighted_sum += weights[j] * prices[i-j]
        result[i] = weighted_sum / weight_sum
    
    return result


@njit(cache=True)
def dema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Double Exponential Moving Average.
    
    DEMA = 2 * EMA(P) - EMA(EMA(P))
    Reduces lag compared to regular EMA.
    """
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


@njit(cache=True)
def tema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Triple Exponential Moving Average.
    
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    Further reduces lag.
    """
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

@njit(cache=True)
def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.
    
    RSI = 100 - 100 / (1 + RS)
    RS = Average Gain / Average Loss
    
    Interpretation:
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    period : int
        RSI period (default 14).
    
    Returns
    -------
    np.ndarray
        RSI values (0-100).
    """
    n = len(prices)
    result = np.empty(n)
    result[:period] = np.nan
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initial average gain/loss (SMA)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    
    # Smoothed RSI (Wilder's smoothing)
    for i in range(period, n-1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i+1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i+1] = 100.0 - 100.0 / (1.0 + rs)
    
    return result


@njit(cache=True)
def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, 
         signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence.
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line
    
    Trading signals:
    - MACD crosses above Signal: Bullish
    - MACD crosses below Signal: Bearish
    - Histogram expansion: Trend strengthening
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).
    
    Returns
    -------
    tuple
        (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@njit(cache=True)
def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.
    
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)
    
    Interpretation:
    - %K > 80: Overbought
    - %K < 20: Oversold
    - %K crosses above %D: Buy signal
    
    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    k_period : int
        Lookback period for %K.
    d_period : int
        Smoothing period for %D.
    
    Returns
    -------
    tuple
        (%K, %D)
    """
    n = len(close)
    k = np.empty(n)
    k[:k_period-1] = np.nan
    
    for i in range(k_period-1, n):
        highest = np.max(high[i-k_period+1:i+1])
        lowest = np.min(low[i-k_period+1:i+1])
        
        if highest == lowest:
            k[i] = 50.0
        else:
            k[i] = (close[i] - lowest) / (highest - lowest) * 100.0
    
    d = sma(k, d_period)
    
    return k, d


@njit(cache=True)
def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """
    Williams %R.
    
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    Range: -100 to 0
    - %R > -20: Overbought
    - %R < -80: Oversold
    """
    n = len(close)
    result = np.empty(n)
    result[:period-1] = np.nan
    
    for i in range(period-1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            result[i] = -50.0
        else:
            result[i] = (highest - close[i]) / (highest - lowest) * -100.0
    
    return result


@njit(cache=True)
def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Price Momentum.
    
    MOM = P_t - P_{t-n}
    
    Positive: Uptrend
    Negative: Downtrend
    """
    n = len(prices)
    result = np.empty(n)
    result[:period] = np.nan
    
    for i in range(period, n):
        result[i] = prices[i] - prices[i-period]
    
    return result


@njit(cache=True)
def roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Rate of Change (percentage).
    
    ROC = (P_t - P_{t-n}) / P_{t-n} * 100
    """
    n = len(prices)
    result = np.empty(n)
    result[:period] = np.nan
    
    for i in range(period, n):
        result[i] = (prices[i] - prices[i-period]) / prices[i-period] * 100.0
    
    return result


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

@njit(cache=True)
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    Average True Range.
    
    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = EMA(TR, period)
    
    Measures volatility - higher ATR = higher volatility.
    
    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    period : int
        ATR period.
    
    Returns
    -------
    np.ndarray
        ATR values.
    """
    n = len(close)
    tr = np.empty(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    return ema(tr, period)


@njit(cache=True)
def bollinger_bands(prices: np.ndarray, period: int = 20, 
                    std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.
    
    Middle = SMA(price, period)
    Upper = Middle + std_dev * StdDev(price, period)
    Lower = Middle - std_dev * StdDev(price, period)
    
    Trading signals:
    - Price touches Upper: Overbought / Sell
    - Price touches Lower: Oversold / Buy
    - Band squeeze (narrow bands): Expect volatility expansion
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    period : int
        SMA period.
    std_dev : float
        Number of standard deviations.
    
    Returns
    -------
    tuple
        (Upper band, Middle band, Lower band)
    """
    n = len(prices)
    middle = sma(prices, period)
    
    upper = np.empty(n)
    lower = np.empty(n)
    upper[:period-1] = np.nan
    lower[:period-1] = np.nan
    
    for i in range(period-1, n):
        std = np.std(prices[i-period+1:i+1])
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
    
    return upper, middle, lower


@njit(cache=True)
def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = 20, atr_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channels.
    
    Middle = EMA(close, period)
    Upper = Middle + atr_mult * ATR
    Lower = Middle - atr_mult * ATR
    
    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    """
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    
    return upper, middle, lower


# =============================================================================
# TREND INDICATORS
# =============================================================================

@njit(cache=True)
def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index.
    
    Measures trend strength (not direction):
    - ADX < 20: Weak trend / Ranging
    - ADX 20-40: Developing trend
    - ADX 40-60: Strong trend
    - ADX > 60: Very strong trend
    
    +DI > -DI: Uptrend
    +DI < -DI: Downtrend
    
    Parameters
    ----------
    high : np.ndarray
        High prices.
    low : np.ndarray
        Low prices.
    close : np.ndarray
        Close prices.
    period : int
        ADX period.
    
    Returns
    -------
    tuple
        (ADX, +DI, -DI)
    """
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    # Smooth with EMA
    atr_smooth = ema(tr, period)
    plus_dm_smooth = ema(plus_dm, period)
    minus_dm_smooth = ema(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(n):
        if atr_smooth[i] != 0:
            plus_di[i] = plus_dm_smooth[i] / atr_smooth[i] * 100
            minus_di[i] = minus_dm_smooth[i] / atr_smooth[i] * 100
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
    
    adx_values = ema(dx, period)
    
    return adx_values, plus_di, minus_di


@njit(cache=True)
def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend Indicator.
    
    Uses ATR to create dynamic support/resistance levels.
    Follows price in the direction of the trend.
    
    Returns
    -------
    tuple
        (Supertrend values, Trend direction: 1=up, -1=down)
    """
    n = len(close)
    atr_values = atr(high, low, close, period)
    
    # Basic bands
    basic_upper = (high + low) / 2 + multiplier * atr_values
    basic_lower = (high + low) / 2 - multiplier * atr_values
    
    # Final bands
    final_upper = np.empty(n)
    final_lower = np.empty(n)
    supertrend = np.empty(n)
    direction = np.empty(n)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    
    for i in range(1, n):
        # Upper band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Lower band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
    
    # Initial direction
    direction[0] = 1 if close[0] > final_upper[0] else -1
    supertrend[0] = final_lower[0] if direction[0] == 1 else final_upper[0]
    
    for i in range(1, n):
        if direction[i-1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, direction


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

@njit(cache=True)
def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.
    
    OBV = prev_OBV + volume (if close > prev_close)
    OBV = prev_OBV - volume (if close < prev_close)
    OBV = prev_OBV (if close == prev_close)
    
    Rising OBV with rising price: Confirms uptrend
    Divergence: Potential trend reversal
    """
    n = len(close)
    result = np.empty(n)
    result[0] = volume[0]
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            result[i] = result[i-1] + volume[i]
        elif close[i] < close[i-1]:
            result[i] = result[i-1] - volume[i]
        else:
            result[i] = result[i-1]
    
    return result


@njit(cache=True)
def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray) -> np.ndarray:
    """
    Volume Weighted Average Price.
    
    VWAP = Σ(Typical Price * Volume) / Σ(Volume)
    Typical Price = (H + L + C) / 3
    
    Used as benchmark - price above VWAP is bullish.
    """
    typical_price = (high + low + close) / 3
    cum_pv = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    
    return cum_pv / cum_vol


@njit(cache=True)
def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        volume: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Money Flow Index.
    
    MFI = 100 - 100 / (1 + Money Flow Ratio)
    Money Flow Ratio = Positive MF / Negative MF
    
    Volume-weighted RSI:
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    """
    n = len(close)
    result = np.empty(n)
    result[:period] = np.nan
    
    # Typical price and raw money flow
    typical = (high + low + close) / 3
    raw_mf = typical * volume
    
    for i in range(period, n):
        pos_mf = 0.0
        neg_mf = 0.0
        
        for j in range(i-period+1, i+1):
            if typical[j] > typical[j-1]:
                pos_mf += raw_mf[j]
            elif typical[j] < typical[j-1]:
                neg_mf += raw_mf[j]
        
        if neg_mf == 0:
            result[i] = 100.0
        else:
            mf_ratio = pos_mf / neg_mf
            result[i] = 100.0 - 100.0 / (1.0 + mf_ratio)
    
    return result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    volume = np.random.randint(1000000, 10000000, n).astype(np.float64)
    
    print("Technical Indicators Test")
    print("=" * 50)
    
    # Moving Averages
    sma20 = sma(close, 20)
    ema20 = ema(close, 20)
    print(f"SMA(20) current: {sma20[-1]:.2f}")
    print(f"EMA(20) current: {ema20[-1]:.2f}")
    
    # RSI
    rsi14 = rsi(close, 14)
    print(f"RSI(14) current: {rsi14[-1]:.2f}")
    
    # MACD
    macd_line, signal_line, hist = macd(close)
    print(f"MACD: {macd_line[-1]:.4f}, Signal: {signal_line[-1]:.4f}")
    
    # Bollinger Bands
    upper, middle, lower = bollinger_bands(close, 20, 2)
    print(f"Bollinger Bands: [{lower[-1]:.2f}, {middle[-1]:.2f}, {upper[-1]:.2f}]")
    
    # ATR
    atr14 = atr(high, low, close, 14)
    print(f"ATR(14): {atr14[-1]:.4f}")
    
    # ADX
    adx14, plus_di, minus_di = adx(high, low, close, 14)
    print(f"ADX(14): {adx14[-1]:.2f}, +DI: {plus_di[-1]:.2f}, -DI: {minus_di[-1]:.2f}")
    
    # OBV
    obv_values = obv(close, volume)
    print(f"OBV current: {obv_values[-1]:,.0f}")
    
    print("\nAll indicators computed successfully!")
