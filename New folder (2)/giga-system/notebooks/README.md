# GIGA SYSTEM — Notebooks

## Research Notebooks

### greek_alpha_analysis.py
Interactive notebook for running the 5-Domain Greek Alpha Framework.

```python
# Example usage (run as script or in Jupyter)
import sys
sys.path.insert(0, "..")

from research.core.alpha_signal_engine import AlphaSignalEngine
from research.core.greek_mathematics import (
    EuclideanOrderSizer, ArchimedeanRebalancer,
    PythagoreanHarmony, ApolloniusCurvature,
    EudoxianConvergence, ZenoConvergence, PlatonicSymmetry
)
import pandas as pd

# Load data
df = pd.read_csv("../data_samples/btc_daily.csv", parse_dates=True, index_col=0)
prices = pd.Series(df['Close'].values)
volumes = pd.Series(df['Volume'].values)

# Run alpha analysis
engine = AlphaSignalEngine()
signal = engine.generate_signal(prices, volumes)
print(f"Signal: {signal.direction} (confidence={signal.confidence:.2f})")
print(f"Report: {engine.get_alpha_report()}")

# Greek mathematics analysis
freqs = PythagoreanHarmony.detect_frequencies(prices)
harmony = PythagoreanHarmony.compute_harmony_score(freqs)
print(f"Pythagorean Harmony: {harmony}")
```

### backtest_validation.py  
Runs walk-forward validation and stores results.

### convergence_proofs.py
Demonstrates Eudoxian convergence proofs for Monte Carlo pricing.
