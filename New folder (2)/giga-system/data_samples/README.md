# GIGA SYSTEM — Sample Data

This directory contains sample market data for testing and development.

## Files
- `btc_daily.csv` — BTC/USD daily OHLCV data (generated or fetched)
- `sample_options.csv` — Sample options chain data for Greeks testing

## Usage
```python
import pandas as pd
df = pd.read_csv("data_samples/btc_daily.csv", parse_dates=True, index_col=0)
```

## Generating Fresh Data
```python
python scripts/fetch_sample_data.py
```
