#!/usr/bin/env bash
# GIGA SYSTEM — Setup Script
# Installs dependencies, verifies structure, runs tests

set -e

echo "============================================"
echo "  GIGA SYSTEM — Setup & Verification"
echo "============================================"

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1] Creating virtual environment..."
    python -m venv venv
fi

# 2. Activate and install
echo "[2] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify imports
echo "[3] Verifying core imports..."
python -c "
from research.core import greeks, black_scholes, monte_carlo, risk_metrics
from research.core.alpha_signal_engine import AlphaSignalEngine
from research.core.greek_mathematics import EuclideanOrderSizer, ArchimedeanRebalancer
from reducer.reducer import DecisionReducer
from backtesting.result_store import BacktestResultStore
print('All core imports verified.')
"

# 4. Run tests
echo "[4] Running tests..."
python -m pytest tests/ -v --tb=short

echo "============================================"
echo "  Setup Complete."
echo "============================================"
