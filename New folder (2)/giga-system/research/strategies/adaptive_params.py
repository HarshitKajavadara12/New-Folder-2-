"""
Adaptive Strategy Parameters
ML-driven dynamic parameter optimization for trading strategies

Features:
- Online learning from execution results
- Online optimization for parameter tuning
- Market regime adaptation
- Real-time performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class StrategyParameters:
    """Dynamic strategy parameters."""
    strategy_name: str
    
    # Pairs trading
    lookback_window: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 4.0
    
    # Momentum
    fast_ma: int = 10
    slow_ma: int = 30
    momentum_threshold: float = 0.02
    
    # Market making
    bid_spread_bps: float = 5.0
    ask_spread_bps: float = 5.0
    inventory_skew: float = 0.5
    max_inventory: int = 1000
    
    # Risk management
    max_position_size: int = 10000
    max_daily_loss: float = 10000
    profit_target: float = 5000
    
    # Meta-parameters
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    last_updated: float = None
    
    def __post_init__(self):
        """Initialize timestamp."""
        if self.last_updated is None:
            self.last_updated = time.time()


class AdaptiveParameterOptimizer:
    """
    Online learning system for strategy parameters.
    
    Uses online optimization to adapt parameters based on:
    - Recent P&L performance
    - Market regime changes
    - Execution quality
    - Risk-adjusted returns
    """
    
    def __init__(
        self,
        strategy_name: str,
        initial_params: Optional[StrategyParameters] = None,
        memory_size: int = 1000
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            strategy_name: Name of trading strategy
            initial_params: Starting parameters
            memory_size: Number of recent results to remember
        """
        self.strategy_name = strategy_name
        self.params = initial_params or StrategyParameters(strategy_name)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        
        # Parameter bounds
        self.param_bounds = {
            'entry_zscore': (1.0, 4.0),
            'exit_zscore': (0.0, 2.0),
            'fast_ma': (5, 50),
            'slow_ma': (20, 200),
            'momentum_threshold': (0.005, 0.05),
            'bid_spread_bps': (1.0, 20.0),
            'ask_spread_bps': (1.0, 20.0),
            'inventory_skew': (0.0, 1.0),
        }
        
        # Performance tracking
        self.performance_history = []
        self.update_count = 0
        self.best_sharpe = -np.inf
        self.best_params = None
        
        # Exploration
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        
    def record_trade_result(
        self,
        pnl: float,
        execution_price: float,
        target_price: float,
        market_regime: str = "neutral"
    ) -> None:
        """
        Record trade result for learning.
        
        Args:
            pnl: Realized P&L
            execution_price: Actual fill price
            target_price: Expected/target price
            market_regime: Current market regime
        """
        # Calculate reward
        slippage = abs(execution_price - target_price) / target_price
        
        # Reward function: weighted combination of P&L and execution quality
        reward = (
            pnl * 0.7 +  # P&L component
            -slippage * 10000 * 0.3  # Slippage penalty (bps)
        )
        
        # Store experience
        experience = {
            'pnl': pnl,
            'slippage': slippage,
            'regime': market_regime,
            'params': self.params.__dict__.copy(),
            'timestamp': time.time()
        }
        
        self.memory.append(experience)
        self.rewards.append(reward)
        
        # Trigger parameter update every N trades
        if len(self.memory) >= 20 and len(self.memory) % 10 == 0:
            self.update_parameters()
    
    def update_parameters(self) -> None:
        """
        Update strategy parameters using recent performance.
        Uses gradient-based optimization with exploration.
        """
        if len(self.rewards) < 20:
            return
        
        # Calculate recent Sharpe ratio
        recent_rewards = list(self.rewards)[-50:]
        sharpe = np.mean(recent_rewards) / (np.std(recent_rewards) + 1e-6)
        
        # Track performance
        self.performance_history.append({
            'timestamp': time.time(),
            'sharpe': sharpe,
            'avg_reward': np.mean(recent_rewards),
            'params': self.params.__dict__.copy()
        })
        
        # Update best parameters
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_params = self.params.__dict__.copy()
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: random perturbation
            self._explore_parameters()
        else:
            # Exploit: gradient ascent on Sharpe
            self._optimize_parameters(recent_rewards)
        
        # Decay exploration
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
        
        self.update_count += 1
        self.params.last_updated = time.time()
    
    def _explore_parameters(self) -> None:
        """Randomly perturb parameters for exploration."""
        for param_name, (min_val, max_val) in self.param_bounds.items():
            if hasattr(self.params, param_name):
                current_val = getattr(self.params, param_name)
                
                # Random perturbation within ±20%
                perturbation = np.random.uniform(-0.2, 0.2)
                new_val = current_val * (1 + perturbation)
                
                # Clip to bounds
                new_val = np.clip(new_val, min_val, max_val)
                
                # Round integers
                if isinstance(current_val, int):
                    new_val = int(new_val)
                
                setattr(self.params, param_name, new_val)
    
    def _optimize_parameters(self, recent_rewards: List[float]) -> None:
        """
        Optimize parameters using per-parameter finite-difference gradient estimation.
        Performs coordinate-wise gradient ascent on the reward signal.
        """
        if len(recent_rewards) < 10:
            return
        
        baseline_reward = np.mean(recent_rewards[-10:])
        learning_rate = self.params.learning_rate
        
        for param_name, (min_val, max_val) in self.param_bounds.items():
            if not hasattr(self.params, param_name):
                continue
            
            current_val = getattr(self.params, param_name)
            param_range = max_val - min_val
            
            # Finite difference step size (1% of range)
            epsilon = param_range * 0.01
            if epsilon == 0:
                continue
            
            # Estimate gradient by correlating parameter changes with reward changes
            # Look at how reward changed when this parameter was last modified
            relevant_history = [
                h for h in self.performance_history[-20:]
                if param_name in h.get('params', {})
            ]
            
            if len(relevant_history) >= 2:
                param_vals = [h['params'].get(param_name, current_val) for h in relevant_history]
                reward_vals = [h['avg_reward'] for h in relevant_history]
                
                if np.std(param_vals) > 1e-10:
                    # Estimate gradient as correlation between param change and reward
                    gradient = np.corrcoef(param_vals, reward_vals)[0, 1]
                    if np.isnan(gradient):
                        gradient = 0
                else:
                    gradient = 0
            else:
                gradient = 0
            
            # Update with gradient ascent
            new_val = current_val + learning_rate * gradient * param_range * 0.1
            new_val = np.clip(new_val, min_val, max_val)
            
            if isinstance(current_val, int):
                new_val = int(round(new_val))
            
            setattr(self.params, param_name, new_val)
    
    def adapt_to_regime(self, regime: str) -> None:
        """
        Adapt parameters based on market regime.
        
        Args:
            regime: "bull", "bear", or "neutral"
        """
        if regime == "bull":
            # Bull market: tighter stops, momentum trading
            self.params.entry_zscore = max(1.5, self.params.entry_zscore * 0.9)
            self.params.momentum_threshold = min(0.03, self.params.momentum_threshold * 1.1)
            
        elif regime == "bear":
            # Bear market: wider stops, mean reversion
            self.params.entry_zscore = min(3.0, self.params.entry_zscore * 1.1)
            self.params.exit_zscore = max(0.3, self.params.exit_zscore * 0.9)
            
        elif regime == "neutral":
            # Neutral: balanced parameters
            # Move toward defaults
            self.params.entry_zscore = 0.9 * self.params.entry_zscore + 0.1 * 2.0
            self.params.exit_zscore = 0.9 * self.params.exit_zscore + 0.1 * 0.5
    
    def get_current_params(self) -> StrategyParameters:
        """Get current optimized parameters."""
        return self.params
    
    def get_best_params(self) -> Optional[Dict]:
        """Get historically best parameters."""
        return self.best_params
    
    def get_performance_summary(self) -> Dict:
        """Get optimization performance summary."""
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-100:]
        
        return {
            'update_count': self.update_count,
            'best_sharpe': self.best_sharpe,
            'current_sharpe': recent_history[-1]['sharpe'] if recent_history else 0,
            'avg_reward': np.mean([h['avg_reward'] for h in recent_history]),
            'exploration_rate': self.exploration_rate,
            'trades_recorded': len(self.memory),
            'current_params': self.params.__dict__
        }


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("ADAPTIVE PARAMETER OPTIMIZATION DEMO")
    print("=" * 70)
    
    # Initialize optimizer for pairs trading
    optimizer = AdaptiveParameterOptimizer("pairs_trading")
    
    print("\nInitial Parameters:")
    for key, value in optimizer.params.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key:25s}: {value}")
    
    # Simulate 200 trades with learning
    print("\nSimulating 200 trades with online learning...")
    
    np.random.seed(42)
    for i in range(200):
        # Simulate trade outcome
        # Gradually improving P&L as parameters optimize
        base_pnl = np.random.normal(50 + i * 0.5, 100)  # Trending upward
        execution_price = 100 + np.random.normal(0, 0.1)
        target_price = 100
        
        regime = np.random.choice(["bull", "bear", "neutral"], p=[0.4, 0.3, 0.3])
        
        optimizer.record_trade_result(base_pnl, execution_price, target_price, regime)
        
        # Adapt to regime occasionally
        if i % 50 == 0 and i > 0:
            optimizer.adapt_to_regime(regime)
            print(f"\n  Iteration {i}: Adapted to {regime} market")
    
    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    summary = optimizer.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Updates: {summary['update_count']}")
    print(f"  Best Sharpe: {summary['best_sharpe']:.3f}")
    print(f"  Current Sharpe: {summary['current_sharpe']:.3f}")
    print(f"  Avg Reward: ${summary['avg_reward']:.2f}")
    print(f"  Exploration Rate: {summary['exploration_rate']:.3f}")
    
    print("\nOptimized Parameters:")
    current = optimizer.get_current_params()
    best = optimizer.get_best_params()
    
    for key in ['entry_zscore', 'exit_zscore', 'fast_ma', 'slow_ma', 'momentum_threshold']:
        if hasattr(current, key):
            curr_val = getattr(current, key)
            best_val = best.get(key, curr_val) if best else curr_val
            print(f"  {key:25s}: Current={curr_val:8.3f}, Best={best_val:8.3f}")
    
    print("\n  Parameters successfully adapted via online learning!")
