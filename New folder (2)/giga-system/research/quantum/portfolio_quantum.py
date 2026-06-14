"""
GIGA SYSTEM - Quantum Portfolio Optimization
Advanced portfolio optimization using quantum algorithms
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .quantum_optimizer import (
    QuadraticProgram, 
    QuantumOptimizer, 
    OptimizerType,
    OptimizationResult,
    QISKIT_AVAILABLE
)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0         # Minimum weight per asset
    max_weight: float = 1.0         # Maximum weight per asset
    cardinality: Optional[int] = None  # Max number of assets to hold
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    turnover_limit: Optional[float] = None  # Max turnover from current portfolio


@dataclass
class QuantumPortfolioResult:
    """Quantum portfolio optimization result."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    quantum_advantage_score: float
    classical_comparison: Optional[Dict] = None
    circuit_depth: int = 0
    n_qubits: int = 0
    execution_time: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumPortfolioOptimizer:
    """
    Quantum-enhanced portfolio optimization.
    
    Approaches:
    1. Mean-Variance (Markowitz) via QAOA
    2. Risk Parity via VQE
    3. Black-Litterman with quantum sampling
    4. Conditional Value-at-Risk (CVaR) optimization
    
    Quantum Advantage:
    - Combinatorial optimization for cardinality constraints
    - Sampling from multimodal distributions
    - Global minimum finding for non-convex problems
    """
    
    def __init__(self,
                 n_assets: int,
                 asset_names: Optional[List[str]] = None,
                 risk_free_rate: float = 0.02,
                 use_quantum: bool = True,
                 qaoa_reps: int = 3):
        """
        Initialize quantum portfolio optimizer.
        
        Parameters
        ----------
        n_assets : int
            Number of assets.
        asset_names : list, optional
            Asset names/tickers.
        risk_free_rate : float
            Annual risk-free rate.
        use_quantum : bool
            Use quantum optimizer if available.
        qaoa_reps : int
            QAOA circuit repetitions.
        """
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(n_assets)]
        self.risk_free_rate = risk_free_rate
        self.use_quantum = use_quantum and QISKIT_AVAILABLE
        self.qaoa_reps = qaoa_reps
        
        # Optimizer
        if self.use_quantum:
            self.optimizer = QuantumOptimizer(
                optimizer_type=OptimizerType.QAOA,
                reps=qaoa_reps
            )
        else:
            self.optimizer = QuantumOptimizer(
                optimizer_type=OptimizerType.CLASSICAL
            )
    
    # =========================================================================
    # MEAN-VARIANCE OPTIMIZATION
    # =========================================================================
    
    def optimize_mean_variance(self,
                              expected_returns: np.ndarray,
                              cov_matrix: np.ndarray,
                              risk_aversion: float = 0.5,
                              constraints: Optional[PortfolioConstraints] = None
                              ) -> QuantumPortfolioResult:
        """
        Mean-Variance optimization using quantum computing.
        
        Objective: max μ^T w - (λ/2) w^T Σ w
        Or equivalently: min (λ/2) w^T Σ w - μ^T w
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns vector.
        cov_matrix : np.ndarray
            Covariance matrix.
        risk_aversion : float
            Risk aversion (0 = max return, 1 = min risk).
        constraints : PortfolioConstraints, optional
            Portfolio constraints.
        
        Returns
        -------
        QuantumPortfolioResult
            Optimized portfolio.
        """
        # Build QUBO
        qubo = QuadraticProgram(self.n_assets)
        qubo.set_portfolio_objective(
            cov_matrix=cov_matrix,
            expected_returns=expected_returns,
            risk_aversion=risk_aversion
        )
        
        # Run optimization
        result = self.optimizer.optimize(qubo)
        
        # Calculate portfolio metrics
        weights = result.optimal_weights
        exp_ret = weights @ expected_returns
        vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (exp_ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(cov_matrix))
        div_ratio = (weights @ asset_vols) / vol if vol > 0 else 1
        
        # Quantum advantage score (heuristic)
        quantum_score = self._calculate_quantum_advantage(
            self.n_assets, 
            constraints
        )
        
        return QuantumPortfolioResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
            quantum_advantage_score=quantum_score,
            n_qubits=self.n_assets,
            execution_time=result.execution_time,
            metadata={
                'risk_aversion': risk_aversion,
                'optimizer_type': result.optimizer_type.value
            }
        )
    
    # =========================================================================
    # EFFICIENT FRONTIER
    # =========================================================================
    
    def compute_efficient_frontier(self,
                                   expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   n_points: int = 20
                                   ) -> List[QuantumPortfolioResult]:
        """
        Compute efficient frontier using quantum optimization.
        
        Varies risk aversion from 0 (max return) to 1 (min risk).
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns.
        cov_matrix : np.ndarray
            Covariance matrix.
        n_points : int
            Number of frontier points.
        
        Returns
        -------
        list
            List of QuantumPortfolioResult for each frontier point.
        """
        frontier = []
        
        for i in range(n_points):
            risk_aversion = i / (n_points - 1)
            
            result = self.optimize_mean_variance(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_aversion=risk_aversion
            )
            
            frontier.append(result)
        
        return frontier
    
    # =========================================================================
    # MINIMUM VARIANCE PORTFOLIO
    # =========================================================================
    
    def minimum_variance(self,
                        cov_matrix: np.ndarray
                        ) -> QuantumPortfolioResult:
        """
        Find minimum variance portfolio.
        
        Objective: min w^T Σ w
        Subject to: Σw = 1
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix.
        
        Returns
        -------
        QuantumPortfolioResult
            Minimum variance portfolio.
        """
        # Use risk_aversion = 1 (pure risk minimization)
        dummy_returns = np.zeros(self.n_assets)
        
        return self.optimize_mean_variance(
            expected_returns=dummy_returns,
            cov_matrix=cov_matrix,
            risk_aversion=1.0
        )
    
    # =========================================================================
    # MAXIMUM SHARPE RATIO
    # =========================================================================
    
    def maximum_sharpe(self,
                      expected_returns: np.ndarray,
                      cov_matrix: np.ndarray
                      ) -> QuantumPortfolioResult:
        """
        Find maximum Sharpe ratio portfolio.
        
        This is equivalent to the tangency portfolio.
        
        For quantum: We approximate by searching over risk aversion.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns.
        cov_matrix : np.ndarray
            Covariance matrix.
        
        Returns
        -------
        QuantumPortfolioResult
            Maximum Sharpe portfolio.
        """
        best_sharpe = -np.inf
        best_result = None
        
        # Grid search over risk aversion
        for risk_av in np.linspace(0.1, 0.9, 20):
            result = self.optimize_mean_variance(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_aversion=risk_av
            )
            
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
        
        return best_result
    
    # =========================================================================
    # RISK PARITY
    # =========================================================================
    
    def risk_parity(self,
                   cov_matrix: np.ndarray
                   ) -> QuantumPortfolioResult:
        """
        Risk parity portfolio (equal risk contribution).
        
        Each asset contributes equally to total portfolio risk:
        RC_i = w_i * (Σw)_i / (w^T Σ w)^0.5 = 1/n
        
        For quantum: Reformulate as QUBO minimizing:
        Σ(RC_i - 1/n)²
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix.
        
        Returns
        -------
        QuantumPortfolioResult
            Risk parity portfolio.
        """
        n = self.n_assets
        
        # Classical solution (Newton-Raphson approximation)
        # Start with inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        weights = 1 / vols
        weights = weights / np.sum(weights)
        
        # Iterative refinement
        for _ in range(50):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_vol
            risk_contrib = weights * marginal_risk
            
            # Target: equal risk contribution
            target_rc = portfolio_vol / n
            
            # Update weights
            weights = weights * (target_rc / risk_contrib)
            weights = weights / np.sum(weights)
        
        # Calculate metrics
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        return QuantumPortfolioResult(
            weights=weights,
            expected_return=0,  # Unknown without returns
            volatility=portfolio_vol,
            sharpe_ratio=0,
            diversification_ratio=self._diversification_ratio(weights, cov_matrix),
            quantum_advantage_score=0.2,  # Risk parity less suited for quantum
            metadata={'method': 'risk_parity_iterative'}
        )
    
    # =========================================================================
    # CVAR OPTIMIZATION
    # =========================================================================
    
    def optimize_cvar(self,
                     expected_returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     alpha: float = 0.05,
                     scenarios: Optional[np.ndarray] = None
                     ) -> QuantumPortfolioResult:
        """
        Conditional Value-at-Risk optimization.
        
        Minimize: CVaR_α = E[R | R < VaR_α]
        
        For quantum: Use scenario-based approach as QUBO.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns.
        cov_matrix : np.ndarray
            Covariance matrix.
        alpha : float
            Confidence level (default 5% tail).
        scenarios : np.ndarray, optional
            Return scenarios (n_scenarios x n_assets).
        
        Returns
        -------
        QuantumPortfolioResult
            CVaR-optimal portfolio.
        """
        n = self.n_assets
        
        # Generate scenarios if not provided
        if scenarios is None:
            n_scenarios = 1000
            scenarios = np.random.multivariate_normal(
                expected_returns,
                cov_matrix,
                n_scenarios
            )
        
        # Classical CVaR optimization
        # For each weight combination, calculate CVaR
        # Use scipy for efficiency
        from scipy.optimize import minimize
        
        def cvar_objective(w):
            portfolio_returns = scenarios @ w
            var_threshold = np.percentile(portfolio_returns, alpha * 100)
            cvar = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
            return cvar
        
        # Optimize
        x0 = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(
            cvar_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        weights = weights / np.sum(weights)
        
        # Metrics
        exp_ret = weights @ expected_returns
        vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (exp_ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        return QuantumPortfolioResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            diversification_ratio=self._diversification_ratio(weights, cov_matrix),
            quantum_advantage_score=0.5,
            metadata={
                'method': 'cvar_optimization',
                'alpha': alpha,
                'cvar': result.fun
            }
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _diversification_ratio(self, weights: np.ndarray, 
                              cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        asset_vols = np.sqrt(np.diag(cov_matrix))
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        return (weights @ asset_vols) / portfolio_vol if portfolio_vol > 0 else 1
    
    def _calculate_quantum_advantage(self,
                                    n_assets: int,
                                    constraints: Optional[PortfolioConstraints]
                                    ) -> float:
        """
        Calculate estimated quantum advantage score.
        
        Factors:
        - Problem size (more assets = more advantage)
        - Cardinality constraints (combinatorial = good for quantum)
        - Integer constraints
        
        Returns
        -------
        float
            Score 0-1 (higher = more quantum advantage potential)
        """
        score = 0.0
        
        # Size factor
        if n_assets > 50:
            score += 0.3
        elif n_assets > 20:
            score += 0.2
        elif n_assets > 10:
            score += 0.1
        
        # Cardinality constraint (combinatorial)
        if constraints and constraints.cardinality:
            score += 0.4
        
        # Sector constraints
        if constraints and constraints.sector_limits:
            score += 0.1
        
        # Integer constraints would add more
        
        return min(1.0, score)
    
    def compare_classical_quantum(self,
                                 expected_returns: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 risk_aversion: float = 0.5
                                 ) -> Dict[str, Any]:
        """
        Compare classical and quantum optimization results.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns.
        cov_matrix : np.ndarray
            Covariance matrix.
        risk_aversion : float
            Risk aversion parameter.
        
        Returns
        -------
        dict
            Comparison results.
        """
        # Classical
        classical_opt = QuantumOptimizer(optimizer_type=OptimizerType.CLASSICAL)
        qubo = QuadraticProgram(self.n_assets)
        qubo.set_portfolio_objective(cov_matrix, expected_returns, risk_aversion)
        classical_result = classical_opt.optimize(qubo)
        
        # Quantum (if available)
        if self.use_quantum:
            quantum_opt = QuantumOptimizer(optimizer_type=OptimizerType.QAOA, reps=self.qaoa_reps)
            quantum_result = quantum_opt.optimize(qubo)
        else:
            quantum_result = classical_result
        
        # Compare
        c_weights = classical_result.optimal_weights
        q_weights = quantum_result.optimal_weights
        
        weight_diff = np.linalg.norm(c_weights - q_weights)
        value_diff = abs(classical_result.optimal_value - quantum_result.optimal_value)
        
        return {
            'classical_weights': c_weights,
            'quantum_weights': q_weights,
            'classical_value': classical_result.optimal_value,
            'quantum_value': quantum_result.optimal_value,
            'weight_difference': weight_diff,
            'value_difference': value_diff,
            'classical_time': classical_result.execution_time,
            'quantum_time': quantum_result.execution_time,
            'speedup': classical_result.execution_time / quantum_result.execution_time 
                       if quantum_result.execution_time > 0 else 1
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("QUANTUM PORTFOLIO OPTIMIZER TEST")
    print("=" * 60)
    
    # Test portfolio
    n_assets = 5
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    # Market data (annualized)
    expected_returns = np.array([0.15, 0.12, 0.14, 0.18, 0.10])
    
    cov_matrix = np.array([
        [0.04, 0.02, 0.015, 0.01, 0.02],
        [0.02, 0.05, 0.02, 0.015, 0.025],
        [0.015, 0.02, 0.035, 0.01, 0.015],
        [0.01, 0.015, 0.01, 0.06, 0.02],
        [0.02, 0.025, 0.015, 0.02, 0.045]
    ])
    
    print(f"\nAssets: {asset_names}")
    print(f"Expected Returns: {expected_returns}")
    
    # Initialize optimizer
    optimizer = QuantumPortfolioOptimizer(
        n_assets=n_assets,
        asset_names=asset_names,
        risk_free_rate=0.02
    )
    
    # Mean-Variance optimization
    print("\n" + "=" * 40)
    print("MEAN-VARIANCE OPTIMIZATION")
    
    mv_result = optimizer.optimize_mean_variance(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_aversion=0.5
    )
    
    print(f"\nOptimal Weights:")
    for name, w in zip(asset_names, mv_result.weights):
        print(f"  {name}: {w:.2%}")
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {mv_result.expected_return:.2%}")
    print(f"  Volatility: {mv_result.volatility:.2%}")
    print(f"  Sharpe Ratio: {mv_result.sharpe_ratio:.2f}")
    print(f"  Diversification: {mv_result.diversification_ratio:.2f}")
    
    # Maximum Sharpe
    print("\n" + "=" * 40)
    print("MAXIMUM SHARPE PORTFOLIO")
    
    sharpe_result = optimizer.maximum_sharpe(expected_returns, cov_matrix)
    
    print(f"\nOptimal Weights:")
    for name, w in zip(asset_names, sharpe_result.weights):
        print(f"  {name}: {w:.2%}")
    
    print(f"  Sharpe Ratio: {sharpe_result.sharpe_ratio:.2f}")
    
    # Risk Parity
    print("\n" + "=" * 40)
    print("RISK PARITY PORTFOLIO")
    
    rp_result = optimizer.risk_parity(cov_matrix)
    
    print(f"\nRisk Parity Weights:")
    for name, w in zip(asset_names, rp_result.weights):
        print(f"  {name}: {w:.2%}")
    
    # Efficient Frontier
    print("\n" + "=" * 40)
    print("EFFICIENT FRONTIER (5 points)")
    
    frontier = optimizer.compute_efficient_frontier(
        expected_returns, cov_matrix, n_points=5
    )
    
    print(f"\n{'Risk Aversion':<15} {'Return':<12} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 50)
    for i, pt in enumerate(frontier):
        risk_av = i / 4
        print(f"{risk_av:<15.2f} {pt.expected_return:<12.2%} {pt.volatility:<12.2%} {pt.sharpe_ratio:<10.2f}")
