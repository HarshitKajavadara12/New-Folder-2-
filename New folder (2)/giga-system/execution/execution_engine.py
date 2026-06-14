"""
Execution Engine
High-frequency fill simulation with realistic market impact

Features:
- Microsecond-level fill simulation
- Market impact modeling
- Partial fill logic
- Realistic slippage calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from enum import Enum


class FillModel(Enum):
    """Fill simulation models."""
    IMMEDIATE = "immediate"  # Instant fills (unrealistic)
    LINEAR = "linear"        # Linear market impact
    SQRT = "sqrt"           # Square-root impact (Almgren-Chriss)
    REALISTIC = "realistic"  # Combines multiple factors
    CHAOTIC = "chaotic"      # Phase 11: Hostile market simulation

@dataclass
class FillResult:
    """Result of fill simulation."""
    filled_quantity: int
    avg_fill_price: float
    total_slippage_bps: float
    market_impact_bps: float
    latency_us: float
    timestamp: float
    partial_fill: bool = False
    status: str = "FILLED"  # FILLED, PARTIAL, REJECTED
    error_message: Optional[str] = None

class ExecutionEngine:
    """
    HFT-level execution engine with realistic fill simulation.
    
    Models:
    - Market impact (temporary + permanent)
    - Liquidity consumption
    - Adverse selection
    - Time-dependent fills
    - Phase 11: Delays, Rejects, Partials
    """
    
    def __init__(
        self,
        fill_model: FillModel = FillModel.REALISTIC,
        base_latency_us: float = 500,
        latency_std_us: float = 100
    ):
        """
        Initialize execution engine.
        
        Args:
            fill_model: Fill simulation model
            base_latency_us: Base execution latency (microseconds)
            latency_std_us: Latency standard deviation
        """
        self.fill_model = fill_model
        self.base_latency_us = base_latency_us
        self.latency_std_us = latency_std_us
        
        # Execution statistics
        self.total_fills = 0
        self.total_volume = 0
        self.total_slippage_bps = 0
        self.avg_latency_us = 0
        
        # Phase 11 Chaos State
        self.chaos_mode = (fill_model == FillModel.CHAOTIC)
        self._last_chaos_event = 0
        
    def execute_order(
        self,
        quantity: int,
        target_price: float,
        market_volume: int = 1000000,
        volatility: float = 0.02,
        urgency: float = 0.5
    ) -> FillResult:
        """
        Execute order with realistic fill simulation.
        
        Args:
            quantity: Order size (shares)
            target_price: Reference price
            market_volume: Average daily volume
            volatility: Asset volatility (annualized)
            urgency: Execution urgency (0=patient, 1=aggressive)
            
        Returns:
            FillResult with execution details
        """
        start_time = time.perf_counter()
        
        # PHASE 11: CHAOS INJECTION
        if self.chaos_mode:
            chaos_result = self._apply_chaos(quantity, target_price)
            if chaos_result:
                return chaos_result

        # Calculate market impact
        impact_bps = self._calculate_market_impact(
            quantity, market_volume, volatility, urgency
        )
        
        # Calculate slippage components
        timing_slippage_bps = self._calculate_timing_slippage(urgency, volatility)
        spread_cost_bps = self._calculate_spread_cost(urgency, volatility)
        
        total_slippage_bps = impact_bps + timing_slippage_bps + spread_cost_bps
        
        # Determine fill quantity (partial fills for large orders)
        filled_qty, partial = self._determine_fill_quantity(
            quantity, market_volume, urgency
        )
        
        # Calculate fill price
        avg_fill_price = target_price * (1 + total_slippage_bps / 10000)
        
        # Simulate execution latency
        latency_us = max(50, np.random.normal(self.base_latency_us, self.latency_std_us))
        
        # Phase 11: Latency Spikes
        if self.chaos_mode and np.random.random() < 0.1: # 10% chance of spike
            latency_us *= np.random.uniform(5, 50) # 5x to 50x spike
        
        # Update statistics
        self.total_fills += 1
        self.total_volume += filled_qty
        self.total_slippage_bps += total_slippage_bps
        self.avg_latency_us = (
            (self.avg_latency_us * (self.total_fills - 1) + latency_us) / self.total_fills
        )
        
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        
        return FillResult(
            filled_quantity=filled_qty,
            avg_fill_price=avg_fill_price,
            total_slippage_bps=total_slippage_bps,
            market_impact_bps=impact_bps,
            latency_us=latency_us,
            timestamp=time.time(),
            partial_fill=partial,
            status="FILLED" if not partial else "PARTIAL"
        )

    def _apply_chaos(self, quantity, price) -> Optional[FillResult]:
        """Phase 11: Simulate hostile exchange behavior"""
        rand = np.random.random()
        
        # 1. Random Rejection (2%)
        if rand < 0.02:
            return FillResult(
                filled_quantity=0,
                avg_fill_price=0,
                total_slippage_bps=0,
                market_impact_bps=0,
                latency_us=1000,
                timestamp=time.time(),
                partial_fill=False,
                status="REJECTED",
                error_message="EXCHANGE_ERROR: Order Limit Exceeded"
            )
            
        # 2. Ghost Order / Timeout (1%)
        if rand < 0.03:
             time.sleep(0.5) # Freeze
             return None # Simulate timeout/disconnect
             
        return None

    def _determine_fill_quantity(
        self,
        quantity: int,
        market_volume: int,
        urgency: float
    ) -> tuple[int, bool]:
        """
        Determine if order fills completely or partially.
        Large orders relative to volume may partially fill.
        Merged implementation (BUG#4 FIX: was defined twice).
        """
        # Chaos mode: 20% chance of partial fill
        if self.chaos_mode:
            rand_val = np.random.random()
            if rand_val < 0.2:
                fill_pct = np.random.uniform(0.1, 0.9)
                return max(1, int(quantity * fill_pct)), True
        
        # Realistic mode: partial fill for large orders relative to volume
        participation = quantity / (market_volume / 390)
        
        if participation < 0.01:  # Small order
            fill_prob = 1.0
        elif participation < 0.05:  # Medium order
            fill_prob = 0.9 + urgency * 0.1
        else:  # Large order
            fill_prob = 0.7 + urgency * 0.2
        
        if np.random.random() < fill_prob:
            return quantity, False
        else:
            # Partial fill (50-90%)
            fill_pct = np.random.uniform(0.5, 0.9)
            filled_qty = int(quantity * fill_pct)
            return max(1, filled_qty), True
    
    def _calculate_market_impact(
        self,
        quantity: int,
        market_volume: int,
        volatility: float,
        urgency: float
    ) -> float:
        """
        Calculate market impact in basis points.
        Uses Almgren-Chriss square-root model with adjustments.
        """
        if self.fill_model == FillModel.IMMEDIATE:
            return 0.0
        
        # Participation rate
        participation = quantity / (market_volume / 390)  # Intraday volume
        
        if self.fill_model == FillModel.LINEAR:
            # Linear impact
            impact_bps = participation * volatility * 10000 * urgency
            
        elif self.fill_model == FillModel.SQRT:
            # Square-root impact (Almgren-Chriss)
            impact_bps = volatility * np.sqrt(participation) * 10000 * urgency
            
        else:  # REALISTIC
            # Combine temporary and permanent impact
            temp_impact = volatility * np.sqrt(participation) * 10000 * urgency
            perm_impact = volatility * participation * 5000  # Smaller permanent
            
            impact_bps = temp_impact + perm_impact
        
        return max(0.1, impact_bps)
    
    def _calculate_timing_slippage(self, urgency: float, volatility: float) -> float:
        """
        Calculate slippage from timing/execution delay.
        Higher urgency = faster execution = less timing slippage.
        """
        # Estimate time to execute (seconds)
        exec_time_s = 1.0 / (urgency + 0.1)  # 0.9s (urgent) to 10s (patient)
        
        # Price drift during execution
        drift_bps = volatility * np.sqrt(exec_time_s / 252 / 6.5) * 10000
        
        # Random walk component
        random_slippage = np.random.normal(0, drift_bps)
        
        return abs(random_slippage)
    
    def _calculate_spread_cost(self, urgency: float, volatility: float) -> float:
        """
        Calculate bid-ask spread cost.
        More urgent orders cross spread, patient orders sit on book.
        """
        # Typical spread (bps) based on volatility
        typical_spread_bps = volatility * 100
        
        # Urgency determines how much spread is crossed
        spread_cost = typical_spread_bps * urgency
        
        return spread_cost
    
    def get_execution_stats(self) -> Dict:
        """Get execution engine statistics."""
        return {
            'total_fills': self.total_fills,
            'total_volume': self.total_volume,
            'avg_slippage_bps': self.total_slippage_bps / max(1, self.total_fills),
            'avg_latency_us': self.avg_latency_us,
            'fill_model': self.fill_model.value
        }


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("EXECUTION ENGINE DEMO")
    print("=" * 70)
    
    # Initialize engine
    engine = ExecutionEngine(
        fill_model=FillModel.REALISTIC,
        base_latency_us=450
    )
    
    print("\nSimulating order execution scenarios...\n")
    
    # Scenario 1: Small order, high urgency
    print("Scenario 1: Small order (1,000 shares), High urgency")
    fill1 = engine.execute_order(
        quantity=1000,
        target_price=100.0,
        market_volume=5000000,
        volatility=0.25,
        urgency=0.9
    )
    print(f"  Filled: {fill1.filled_quantity:,} shares @ ${fill1.avg_fill_price:.4f}")
    print(f"  Total slippage: {fill1.total_slippage_bps:.2f} bps")
    print(f"  Market impact: {fill1.market_impact_bps:.2f} bps")
    print(f"  Latency: {fill1.latency_us:.1f} μs")
    print(f"  Partial fill: {fill1.partial_fill}")
    
    # Scenario 2: Large order, medium urgency
    print("\nScenario 2: Large order (50,000 shares), Medium urgency")
    fill2 = engine.execute_order(
        quantity=50000,
        target_price=100.0,
        market_volume=2000000,
        volatility=0.30,
        urgency=0.5
    )
    print(f"  Filled: {fill2.filled_quantity:,} shares @ ${fill2.avg_fill_price:.4f}")
    print(f"  Total slippage: {fill2.total_slippage_bps:.2f} bps")
    print(f"  Market impact: {fill2.market_impact_bps:.2f} bps")
    print(f"  Latency: {fill2.latency_us:.1f} μs")
    print(f"  Partial fill: {fill2.partial_fill}")
    
    # Scenario 3: Mega order, low urgency
    print("\nScenario 3: Mega order (100,000 shares), Low urgency")
    fill3 = engine.execute_order(
        quantity=100000,
        target_price=100.0,
        market_volume=1000000,
        volatility=0.40,
        urgency=0.2
    )
    print(f"  Filled: {fill3.filled_quantity:,} shares @ ${fill3.avg_fill_price:.4f}")
    print(f"  Total slippage: {fill3.total_slippage_bps:.2f} bps")
    print(f"  Market impact: {fill3.market_impact_bps:.2f} bps")
    print(f"  Latency: {fill3.latency_us:.1f} μs")
    print(f"  Partial fill: {fill3.partial_fill}")
    
    # Display engine statistics
    print("\n" + "=" * 70)
    print("ENGINE STATISTICS")
    print("=" * 70)
    
    stats = engine.get_execution_stats()
    print(f"Total fills: {stats['total_fills']}")
    print(f"Total volume: {stats['total_volume']:,} shares")
    print(f"Avg slippage: {stats['avg_slippage_bps']:.2f} bps")
    print(f"Avg latency: {stats['avg_latency_us']:.1f} μs")
    print(f"Fill model: {stats['fill_model']}")
    
    print("\n  Execution engine operational!")
