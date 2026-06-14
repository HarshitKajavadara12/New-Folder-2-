
import yfinance as yf
try:
    ticker = yf.Ticker("BTC-USD")
    # Try different methods to get price quickly
    price = ticker.fast_info.last_price
    print(f"SUCCESS: {price}")
except Exception as e:
    print(f"ERROR: {e}")
