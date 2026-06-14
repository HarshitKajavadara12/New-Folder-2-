#!/usr/bin/env python3
"""
Generate sample market data for testing.
Produces realistic synthetic data when real data is unavailable.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_btc_daily(n_days=730, start_price=30000.0):
    """Generate realistic BTC-like daily OHLCV data."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    # Simulate GBM with regime switches
    returns = np.random.normal(0.0005, 0.03, n_days)
    # Add crash regime
    returns[300:320] = np.random.normal(-0.02, 0.06, 20)
    # Add bull regime
    returns[500:600] = np.random.normal(0.003, 0.02, 100)

    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_price = np.roll(close, 1)
    open_price[0] = start_price
    volume = np.random.lognormal(15, 0.5, n_days)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def generate_sample_options(underlying_price=50000.0, n_strikes=20):
    """Generate sample options chain data."""
    strikes = np.linspace(
        underlying_price * 0.7, underlying_price * 1.3, n_strikes
    )
    rows = []
    for strike in strikes:
        for opt_type in ["call", "put"]:
            moneyness = underlying_price / strike
            iv = 0.5 + 0.3 * abs(1 - moneyness)  # Smile
            rows.append(
                {
                    "strike": strike,
                    "type": opt_type,
                    "expiry_days": 30,
                    "iv": iv,
                    "underlying": underlying_price,
                    "bid": max(0, (underlying_price - strike if opt_type == "call" else strike - underlying_price)) * 0.95,
                    "ask": max(0.01, (underlying_price - strike if opt_type == "call" else strike - underlying_price)) * 1.05 + 50,
                    "volume": int(np.random.lognormal(5, 1)),
                    "open_interest": int(np.random.lognormal(7, 1)),
                }
            )
    return pd.DataFrame(rows)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_samples")
    os.makedirs(out_dir, exist_ok=True)

    # BTC daily
    btc = generate_btc_daily()
    btc_path = os.path.join(out_dir, "btc_daily.csv")
    btc.to_csv(btc_path)
    print(f"Wrote {len(btc)} rows to {btc_path}")

    # Options chain
    opts = generate_sample_options()
    opts_path = os.path.join(out_dir, "sample_options.csv")
    opts.to_csv(opts_path, index=False)
    print(f"Wrote {len(opts)} rows to {opts_path}")


if __name__ == "__main__":
    main()
