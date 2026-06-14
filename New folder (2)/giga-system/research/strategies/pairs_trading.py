"""
GIGA SYSTEM - Pairs Trading Strategy
Statistical arbitrage based on cointegration and mean reversion
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats

from .base import Strategy, Signal, Side, Position


@dataclass
class PairStats:
    """Statistics for a trading pair."""
    symbol1: str
    symbol2: str
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float
    correlation: float
    cointegration_pvalue: float
    is_cointegrated: bool


class PairsTradingStrategy(Strategy):
    """
    Statistical Arbitrage Pairs Trading Strategy.
    
    Concept:
    - Find two cointegrated assets (spread is mean-reverting)
    - Go long spread when it's below mean (buy A, sell B)
    - Go short spread when it's above mean (sell A, buy B)
    
    Mathematical Foundation:
    - Engle-Granger two-step cointegration test
    - Ornstein-Uhlenbeck process for spread dynamics
    - Half-life calculation for mean reversion speed
    
    Entry/Exit Rules:
    - Entry: |Z-score| > entry_threshold (default 2.0)
    - Exit: Z-score crosses 0 or hits exit threshold
    - Stop loss: |Z-score| > stop_loss_threshold
    
    Position Sizing:
    - Dollar neutral: $1 long = $1 short
    - Hedge ratio adjusted for optimal spread
    """
    
    def __init__(self, 
                 symbol1: str, 
                 symbol2: str,
                 lookback: int = 60,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.0,
                 stop_zscore: float = 4.0,
                 min_half_life: int = 5,
                 max_half_life: int = 60,
                 recalculate_interval: int = 20):
        """
        Initialize pairs trading strategy.
        
        Parameters
        ----------
        symbol1 : str
            First symbol (typically the one to go long).
        symbol2 : str
            Second symbol (hedge).
        lookback : int
            Lookback period for statistics (default 60).
        entry_zscore : float
            Z-score threshold for entry (default 2.0).
        exit_zscore : float
            Z-score threshold for exit (default 0.0 = mean).
        stop_zscore : float
            Z-score threshold for stop loss (default 4.0).
        min_half_life : int
            Minimum acceptable half-life in periods.
        max_half_life : int
            Maximum acceptable half-life in periods.
        recalculate_interval : int
            Periods between hedge ratio recalculation.
        """
        super().__init__(
            name="Pairs_Trading",
            symbols=[symbol1, symbol2],
            lookback=lookback,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_zscore=stop_zscore
        )
        
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.recalculate_interval = recalculate_interval
        
        # State
        self._prices1: List[float] = []
        self._prices2: List[float] = []
        self._pair_stats: Optional[PairStats] = None
        self._bars_since_recalc = 0
        self._in_position = False
        self._position_side = 0  # 1 = long spread, -1 = short spread
    
    def initialize(self, **kwargs):
        """Initialize strategy state."""
        super().initialize(**kwargs)
        self._prices1 = []
        self._prices2 = []
        self._pair_stats = None
        self._bars_since_recalc = 0
    
    # =========================================================================
    # COINTEGRATION ANALYSIS
    # =========================================================================
    
    def calculate_hedge_ratio(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Calculate hedge ratio using OLS regression.
        
        y = β * x + ε
        β = cov(y, x) / var(x)
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable (symbol1 prices).
        x : np.ndarray
            Independent variable (symbol2 prices).
        
        Returns
        -------
        float
            Hedge ratio (β).
        """
        # OLS regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 1.0
        
        return numerator / denominator
    
    def calculate_spread(self, prices1: np.ndarray, prices2: np.ndarray,
                        hedge_ratio: float) -> np.ndarray:
        """
        Calculate spread between two price series.
        
        Spread = P1 - β * P2
        
        Parameters
        ----------
        prices1 : np.ndarray
            First price series.
        prices2 : np.ndarray
            Second price series.
        hedge_ratio : float
            Hedge ratio.
        
        Returns
        -------
        np.ndarray
            Spread series.
        """
        return prices1 - hedge_ratio * prices2
    
    def test_cointegration(self, y: np.ndarray, x: np.ndarray) -> Tuple[float, bool]:
        """
        Test for cointegration using Engle-Granger method.
        
        Steps:
        1. Regress y on x
        2. Test residuals for stationarity (ADF test)
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable.
        x : np.ndarray
            Independent variable.
        
        Returns
        -------
        tuple
            (p-value, is_cointegrated)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Calculate residuals
            hedge_ratio = self.calculate_hedge_ratio(y, x)
            spread = self.calculate_spread(y, x, hedge_ratio)
            
            # ADF test on spread
            adf_result = adfuller(spread, maxlag=1)
            pvalue = adf_result[1]
            
            # Cointegrated if p-value < 0.05
            return pvalue, pvalue < 0.05
            
        except ImportError:
            # Fallback: simple correlation-based heuristic
            corr = np.corrcoef(y, x)[0, 1]
            # High correlation suggests potential cointegration
            return 0.01 if abs(corr) > 0.8 else 0.5, abs(corr) > 0.8
    
    def calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion.
        
        Using Ornstein-Uhlenbeck process:
        dS = θ(μ - S)dt + σdW
        
        Half-life = ln(2) / θ
        
        θ is estimated from: ΔS_t = θ * S_{t-1} + ε
        
        Parameters
        ----------
        spread : np.ndarray
            Spread series.
        
        Returns
        -------
        float
            Half-life in periods.
        """
        # Lagged spread
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        
        # Regression: ΔS = θ * S_lag + ε
        # θ should be negative for mean reversion
        x = spread_lag.reshape(-1, 1)
        y = spread_diff
        
        # OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        theta = np.sum((x.flatten() - x_mean) * (y - y_mean)) / np.sum((x.flatten() - x_mean) ** 2)
        
        if theta >= 0:
            # Not mean reverting
            return float('inf')
        
        half_life = -np.log(2) / theta
        
        return max(1.0, half_life)
    
    def calculate_pair_stats(self) -> Optional[PairStats]:
        """
        Calculate all statistics for the pair.
        
        Returns
        -------
        PairStats or None
            Pair statistics if enough data.
        """
        if len(self._prices1) < self.lookback:
            return None
        
        # Use most recent lookback period
        p1 = np.array(self._prices1[-self.lookback:])
        p2 = np.array(self._prices2[-self.lookback:])
        
        # Hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(p1, p2)
        
        # Spread
        spread = self.calculate_spread(p1, p2, hedge_ratio)
        
        # Spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Correlation
        correlation = np.corrcoef(p1, p2)[0, 1]
        
        # Cointegration test
        pvalue, is_cointegrated = self.test_cointegration(p1, p2)
        
        # Half-life
        half_life = self.calculate_half_life(spread)
        
        return PairStats(
            symbol1=self.symbol1,
            symbol2=self.symbol2,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            correlation=correlation,
            cointegration_pvalue=pvalue,
            is_cointegrated=is_cointegrated
        )
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def calculate_zscore(self) -> float:
        """
        Calculate current Z-score of spread.
        
        Z = (spread - mean) / std
        
        Returns
        -------
        float
            Current Z-score.
        """
        if self._pair_stats is None:
            return 0.0
        
        current_spread = (self._prices1[-1] - 
                         self._pair_stats.hedge_ratio * self._prices2[-1])
        
        zscore = ((current_spread - self._pair_stats.spread_mean) / 
                  self._pair_stats.spread_std)
        
        return zscore
    
    def should_recalculate(self) -> bool:
        """Check if hedge ratio should be recalculated."""
        return (self._pair_stats is None or 
                self._bars_since_recalc >= self.recalculate_interval)
    
    def is_valid_pair(self) -> bool:
        """
        Check if pair meets trading criteria.
        
        Criteria:
        1. Cointegrated (p-value < 0.05)
        2. Half-life within acceptable range
        3. Sufficient correlation
        """
        if self._pair_stats is None:
            return False
        
        stats = self._pair_stats
        
        # Check cointegration
        if not stats.is_cointegrated:
            return False
        
        # Check half-life
        if stats.half_life < self.min_half_life:
            return False
        if stats.half_life > self.max_half_life:
            return False
        
        # Check correlation
        if abs(stats.correlation) < 0.5:
            return False
        
        return True
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """
        Process new bar and generate signals.
        
        Parameters
        ----------
        timestamp : datetime
            Bar timestamp.
        bars : dict
            Dictionary of symbol -> OHLCV data.
        
        Returns
        -------
        list
            List of Signal objects.
        """
        # Update price history
        if self.symbol1 in bars and self.symbol2 in bars:
            self._prices1.append(bars[self.symbol1]['close'])
            self._prices2.append(bars[self.symbol2]['close'])
            self._bars_since_recalc += 1
        else:
            return []
        
        # Trim history
        if len(self._prices1) > self.lookback * 2:
            self._prices1 = self._prices1[-self.lookback * 2:]
            self._prices2 = self._prices2[-self.lookback * 2:]
        
        # Not enough data
        if len(self._prices1) < self.lookback:
            return []
        
        # Recalculate statistics if needed
        if self.should_recalculate():
            self._pair_stats = self.calculate_pair_stats()
            self._bars_since_recalc = 0
        
        # Check if pair is valid for trading
        if not self.is_valid_pair():
            return []
        
        # Calculate current Z-score
        zscore = self.calculate_zscore()
        
        signals = []
        
        # Entry signals
        if not self._in_position:
            if zscore > self.entry_zscore:
                # Spread is high: short spread (sell sym1, buy sym2)
                signals.extend(self._generate_short_spread_signals(timestamp, zscore))
                self._in_position = True
                self._position_side = -1
                
            elif zscore < -self.entry_zscore:
                # Spread is low: long spread (buy sym1, sell sym2)
                signals.extend(self._generate_long_spread_signals(timestamp, zscore))
                self._in_position = True
                self._position_side = 1
        
        # Exit signals
        else:
            should_exit = False
            exit_reason = ""
            
            # Mean reversion exit
            if self._position_side == 1 and zscore >= self.exit_zscore:
                should_exit = True
                exit_reason = "mean_reversion"
            elif self._position_side == -1 and zscore <= self.exit_zscore:
                should_exit = True
                exit_reason = "mean_reversion"
            
            # Stop loss
            if abs(zscore) > self.stop_zscore:
                should_exit = True
                exit_reason = "stop_loss"
            
            if should_exit:
                signals.extend(self._generate_exit_signals(timestamp, zscore, exit_reason))
                self._in_position = False
                self._position_side = 0
        
        # Store signals
        self._signals.extend(signals)
        
        return signals
    
    def _generate_long_spread_signals(self, timestamp: datetime, 
                                       zscore: float) -> List[Signal]:
        """Generate signals for going long the spread."""
        return [
            Signal(
                timestamp=timestamp,
                symbol=self.symbol1,
                side=Side.BUY,
                strength=abs(zscore) / self.entry_zscore,
                confidence=1.0 - self._pair_stats.cointegration_pvalue,
                metadata={
                    'strategy': 'pairs_trading',
                    'action': 'long_spread',
                    'zscore': zscore,
                    'hedge_ratio': self._pair_stats.hedge_ratio,
                    'half_life': self._pair_stats.half_life
                }
            ),
            Signal(
                timestamp=timestamp,
                symbol=self.symbol2,
                side=Side.SELL,
                strength=abs(zscore) / self.entry_zscore,
                confidence=1.0 - self._pair_stats.cointegration_pvalue,
                target_position=-self._pair_stats.hedge_ratio,  # Relative to symbol1
                metadata={
                    'strategy': 'pairs_trading',
                    'action': 'long_spread',
                    'zscore': zscore,
                    'hedge_ratio': self._pair_stats.hedge_ratio
                }
            )
        ]
    
    def _generate_short_spread_signals(self, timestamp: datetime,
                                        zscore: float) -> List[Signal]:
        """Generate signals for going short the spread."""
        return [
            Signal(
                timestamp=timestamp,
                symbol=self.symbol1,
                side=Side.SELL,
                strength=abs(zscore) / self.entry_zscore,
                confidence=1.0 - self._pair_stats.cointegration_pvalue,
                metadata={
                    'strategy': 'pairs_trading',
                    'action': 'short_spread',
                    'zscore': zscore,
                    'hedge_ratio': self._pair_stats.hedge_ratio
                }
            ),
            Signal(
                timestamp=timestamp,
                symbol=self.symbol2,
                side=Side.BUY,
                strength=abs(zscore) / self.entry_zscore,
                confidence=1.0 - self._pair_stats.cointegration_pvalue,
                target_position=self._pair_stats.hedge_ratio,
                metadata={
                    'strategy': 'pairs_trading',
                    'action': 'short_spread',
                    'zscore': zscore
                }
            )
        ]
    
    def _generate_exit_signals(self, timestamp: datetime, zscore: float,
                               reason: str) -> List[Signal]:
        """Generate exit signals."""
        # Close both positions
        if self._position_side == 1:
            # Was long spread: sell sym1, buy sym2
            return [
                Signal(
                    timestamp=timestamp,
                    symbol=self.symbol1,
                    side=Side.SELL,
                    strength=0.0,
                    confidence=1.0,
                    metadata={'strategy': 'pairs_trading', 'action': 'exit', 
                             'reason': reason, 'zscore': zscore}
                ),
                Signal(
                    timestamp=timestamp,
                    symbol=self.symbol2,
                    side=Side.BUY,
                    strength=0.0,
                    confidence=1.0,
                    metadata={'strategy': 'pairs_trading', 'action': 'exit',
                             'reason': reason}
                )
            ]
        else:
            # Was short spread: buy sym1, sell sym2
            return [
                Signal(
                    timestamp=timestamp,
                    symbol=self.symbol1,
                    side=Side.BUY,
                    strength=0.0,
                    confidence=1.0,
                    metadata={'strategy': 'pairs_trading', 'action': 'exit',
                             'reason': reason, 'zscore': zscore}
                ),
                Signal(
                    timestamp=timestamp,
                    symbol=self.symbol2,
                    side=Side.SELL,
                    strength=0.0,
                    confidence=1.0,
                    metadata={'strategy': 'pairs_trading', 'action': 'exit',
                             'reason': reason}
                )
            ]
    
    def finalize(self) -> Dict[str, Any]:
        """Return strategy summary."""
        base_summary = super().finalize()
        
        # Add pairs-specific metrics
        if self._pair_stats:
            base_summary.update({
                'hedge_ratio': self._pair_stats.hedge_ratio,
                'half_life': self._pair_stats.half_life,
                'correlation': self._pair_stats.correlation,
                'cointegration_pvalue': self._pair_stats.cointegration_pvalue,
                'is_cointegrated': self._pair_stats.is_cointegrated
            })
        
        return base_summary


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta
    
    # Use REAL cointegrated pair (e.g., PEP/KO, GM/F, etc.)
    try:
        from data.realtime_manager import get_data_manager
        dm = get_data_manager()
        
        # Coca-Cola and PepsiCo are often cointegrated
        ko_df = dm.get_historical_data_sync('KO', '2023-01-01', '2023-12-31')
        pep_df = dm.get_historical_data_sync('PEP', '2023-01-01', '2023-12-31')
        
        if not ko_df.empty and not pep_df.empty:
            print("  Using REAL pair: KO vs PEP")
            
            # Align data
            min_len = min(len(ko_df), len(pep_df))
            prices1 = ko_df['close'].values[:min_len]
            prices2 = pep_df['close'].values[:min_len]
            n = min_len
        else:
            raise Exception("No data")
    except Exception as e:
        print(f"  Real pair data unavailable: {e}")
        print("  Pairs trading demonstration requires KO and PEP historical data")
        import sys
        sys.exit(0)
    
    # Create strategy
    strategy = PairsTradingStrategy(
        symbol1="STOCK_A",
        symbol2="STOCK_B",
        lookback=60,
        entry_zscore=2.0,
        exit_zscore=0.0
    )
    strategy.initialize()
    
    # Simulate
    signals = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n):
        timestamp = base_date + timedelta(days=i)
        bars = {
            "STOCK_A": {'open': prices1[i], 'high': prices1[i]*1.01, 
                       'low': prices1[i]*0.99, 'close': prices1[i], 'volume': 1000000},
            "STOCK_B": {'open': prices2[i], 'high': prices2[i]*1.01,
                       'low': prices2[i]*0.99, 'close': prices2[i], 'volume': 1000000}
        }
        
        new_signals = strategy.on_bar(timestamp, bars)
        signals.extend(new_signals)
    
    # Summary
    summary = strategy.finalize()
    
    print("Pairs Trading Strategy Test")
    print("=" * 50)
    print(f"Total signals generated: {len(signals)}")
    print(f"Hedge ratio: {summary.get('hedge_ratio', 'N/A'):.4f}")
    print(f"Half-life: {summary.get('half_life', 'N/A'):.1f} days")
    print(f"Correlation: {summary.get('correlation', 'N/A'):.4f}")
    print(f"Cointegrated: {summary.get('is_cointegrated', 'N/A')}")
    
    # Count entries/exits
    entries = [s for s in signals if 'entry' in str(s.metadata.get('action', '')) 
               or 'spread' in str(s.metadata.get('action', ''))]
    exits = [s for s in signals if s.metadata.get('action') == 'exit']
    
    print(f"Entry signals: {len(entries)}")
    print(f"Exit signals: {len(exits)}")
