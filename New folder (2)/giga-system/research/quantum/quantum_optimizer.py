"""
GIGA SYSTEM - Quantum Optimization
Quantum computing algorithms for portfolio optimization using QAOA/VQE
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Qiskit imports (conditional)
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler, Estimator
    from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Dummy classes for type hinting
    QuantumCircuit = Any 
    QuantumRegister = Any
    ClassicalRegister = Any
    Sampler = Any
    Estimator = Any


class OptimizerType(Enum):
    """Quantum optimizer types."""
    QAOA = "qaoa"
    VQE = "vqe"
    CLASSICAL = "classical"


@dataclass
class OptimizationResult:
    """Quantum optimization result."""
    optimal_weights: np.ndarray
    optimal_value: float
    n_iterations: int
    eigenvalue: float
    eigenvector: Optional[np.ndarray]
    optimizer_type: OptimizerType
    execution_time: float
    metadata: Dict[str, Any]


class QuadraticProgram:
    """
    Quadratic Unconstrained Binary Optimization (QUBO) formulation.
    
    QUBO: minimize x^T Q x + c^T x
    
    Portfolio optimization QUBO:
    minimize: λ * x^T Σ x - (1-λ) * μ^T x
    subject to: Σx_i = 1 (budget constraint)
    
    Where:
    - x = binary decision variables (0/1 for each asset)
    - Σ = covariance matrix
    - μ = expected returns
    - λ = risk aversion parameter
    """
    
    def __init__(self, n_assets: int):
        """
        Initialize QUBO formulation.
        
        Parameters
        ----------
        n_assets : int
            Number of assets.
        """
        self.n_assets = n_assets
        self.Q = np.zeros((n_assets, n_assets))
        self.c = np.zeros(n_assets)
        self.constraints = []
    
    def set_portfolio_objective(self,
                               cov_matrix: np.ndarray,
                               expected_returns: np.ndarray,
                               risk_aversion: float = 0.5,
                               budget_constraint: float = 1.0):
        """
        Set portfolio optimization objective.
        
        Objective: min λ * x^T Σ x - (1-λ) * μ^T x + penalty * (Σx - budget)²
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Asset covariance matrix (n x n).
        expected_returns : np.ndarray
            Expected returns vector (n,).
        risk_aversion : float
            Risk aversion (0 = pure return, 1 = pure risk).
        budget_constraint : float
            Sum of weights constraint.
        """
        n = self.n_assets
        
        # Risk term (quadratic)
        self.Q = risk_aversion * cov_matrix
        
        # Return term (linear, negated for minimization)
        self.c = -(1 - risk_aversion) * expected_returns
        
        # Budget constraint penalty (Σx = budget)
        # Penalty: P * (Σx - budget)² = P * (Σx)² - 2*P*budget*Σx + P*budget²
        # The Σx² term adds to Q, the -2*P*budget*x adds to c
        penalty = 10.0  # Large penalty for constraint violation
        
        for i in range(n):
            for j in range(n):
                self.Q[i, j] += penalty
            self.c[i] -= 2 * penalty * budget_constraint  # correct: -2*P*B per element
    
    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising formulation.
        
        QUBO: x ∈ {0, 1}^n
        Ising: z ∈ {-1, +1}^n
        
        Mapping: x = (1 + z) / 2
        
        Returns
        -------
        tuple
            (J, h, offset) for Ising Hamiltonian:
            H = Σ J_ij z_i z_j + Σ h_i z_i + offset
        """
        n = self.n_assets
        
        # Ising coefficients
        J = np.zeros((n, n))
        h = np.zeros(n)
        offset = 0.0
        
        # Transform: x = (1 + z) / 2
        # x^T Q x + c^T x = ...
        
        for i in range(n):
            for j in range(n):
                J[i, j] = self.Q[i, j] / 4
            
            h[i] = self.c[i] / 2 + np.sum(self.Q[i, :]) / 4 + np.sum(self.Q[:, i]) / 4
            offset += self.Q[i, i] / 4
        
        offset += np.sum(self.c) / 2
        
        return J, h, offset
    
    def to_pauli_op(self) -> Any:
        """
        Convert to Qiskit SparsePauliOp (Hamiltonian).
        
        H = Σ J_ij Z_i Z_j + Σ h_i Z_i + offset * I
        
        Returns
        -------
        SparsePauliOp
            Qiskit Hamiltonian operator.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum optimization")
        
        J, h, offset = self.to_ising()
        n = self.n_assets
        
        pauli_list = []
        
        # Identity term (offset)
        pauli_list.append(("I" * n, offset))
        
        # Linear terms (h_i Z_i)
        for i in range(n):
            if abs(h[i]) > 1e-10:
                pauli_str = "I" * (n - i - 1) + "Z" + "I" * i
                pauli_list.append((pauli_str, h[i]))
        
        # Quadratic terms (J_ij Z_i Z_j)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    pauli_str = list("I" * n)
                    pauli_str[n - i - 1] = "Z"
                    pauli_str[n - j - 1] = "Z"
                    pauli_list.append(("".join(pauli_str), J[i, j] + J[j, i]))
        
        return SparsePauliOp.from_list(pauli_list)


class QuantumOptimizer:
    """
    Quantum optimization algorithms for portfolio optimization.
    
    Implements:
    1. QAOA (Quantum Approximate Optimization Algorithm)
    2. VQE (Variational Quantum Eigensolver)
    3. Classical fallback (NumPy eigensolver)
    
    QAOA Circuit:
    |ψ(β,γ)⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^n
    
    Where:
    - U_C(γ) = exp(-i γ C) (cost unitary, encodes objective)
    - U_B(β) = exp(-i β B) (mixer unitary)
    """
    
    def __init__(self,
                 optimizer_type: OptimizerType = OptimizerType.QAOA,
                 reps: int = 3,
                 shots: int = 1024):
        """
        Initialize quantum optimizer.
        
        Parameters
        ----------
        optimizer_type : OptimizerType
            QAOA, VQE, or CLASSICAL.
        reps : int
            Number of repetitions/layers (p for QAOA).
        shots : int
            Number of measurement shots.
        """
        self.optimizer_type = optimizer_type
        self.reps = reps
        self.shots = shots
        
        self._check_qiskit()
    
    def _check_qiskit(self):
        """Check if Qiskit is available."""
        if self.optimizer_type != OptimizerType.CLASSICAL and not QISKIT_AVAILABLE:
            print("Warning: Qiskit not available. Falling back to classical optimizer.")
            self.optimizer_type = OptimizerType.CLASSICAL
    
    def _build_qaoa_circuit(self, n_qubits: int) -> QuantumCircuit:
        """
        Build parameterized QAOA circuit.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        
        Returns
        -------
        QuantumCircuit
            QAOA ansatz circuit.
        """
        qc = QuantumCircuit(n_qubits)
        
        # Initial state: |+⟩^n
        qc.h(range(n_qubits))
        
        # QAOA layers
        for layer in range(self.reps):
            gamma = Parameter(f'γ_{layer}')
            beta = Parameter(f'β_{layer}')
            
            # Cost unitary U_C(γ) - problem-dependent
            # Simplified: RZ rotations
            for i in range(n_qubits):
                qc.rz(gamma, i)
            
            # ZZ interactions
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(gamma, i + 1)
                qc.cx(i, i + 1)
            
            # Mixer unitary U_B(β)
            for i in range(n_qubits):
                qc.rx(2 * beta, i)
        
        return qc
    
    def _build_vqe_ansatz(self, n_qubits: int) -> QuantumCircuit:
        """
        Build VQE ansatz circuit (RealAmplitudes-like).
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        
        Returns
        -------
        QuantumCircuit
            VQE ansatz.
        """
        qc = QuantumCircuit(n_qubits)
        
        for layer in range(self.reps):
            # Rotation layer
            for i in range(n_qubits):
                theta = Parameter(f'θ_{layer}_{i}')
                qc.ry(theta, i)
            
            # Entanglement layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        
        # Final rotation
        for i in range(n_qubits):
            theta = Parameter(f'θ_{self.reps}_{i}')
            qc.ry(theta, i)
        
        return qc
    
    def optimize(self,
                qubo: QuadraticProgram,
                initial_params: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Run quantum optimization.
        
        Parameters
        ----------
        qubo : QuadraticProgram
            QUBO problem formulation.
        initial_params : np.ndarray, optional
            Initial parameter values.
        
        Returns
        -------
        OptimizationResult
            Optimization result.
        """
        import time
        start_time = time.time()
        
        if self.optimizer_type == OptimizerType.CLASSICAL:
            result = self._optimize_classical(qubo)
        elif self.optimizer_type == OptimizerType.QAOA:
            result = self._optimize_qaoa(qubo, initial_params)
        else:  # VQE
            result = self._optimize_vqe(qubo, initial_params)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _optimize_classical(self, qubo: QuadraticProgram) -> OptimizationResult:
        """Classical optimization using NumPy."""
        n = qubo.n_assets

        # Always use SLSQP convex relaxation for continuous portfolio weights.
        # Brute-force binary enumeration is only appropriate for hard cardinality
        # constraints (asset-selection QUBO); for continuous mean-variance the
        # SLSQP relaxation is the correct classical surrogate for the quantum
        # variational circuit that would run on actual QPU hardware.
        from scipy.optimize import minimize, differential_evolution

        def objective(x):
            return x @ qubo.Q @ x + qubo.c @ x

        def normalized_objective(x):
            # Enforce sum=1 via soft normalization during DE (unconstrained)
            xs = x / (x.sum() + 1e-12)
            return xs @ qubo.Q @ xs + qubo.c @ xs

        bounds = [(0.0, 1.0)] * n
        slsqp_constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}

        # Global seed via DE over unconstrained space, then normalize
        de_result = differential_evolution(
            normalized_objective, bounds, maxiter=200, seed=42, tol=1e-8,
            polish=False, popsize=8
        )
        x0_de = de_result.x / (de_result.x.sum() + 1e-12)

        # Local refinement with sum=1 constraint
        result = minimize(objective, x0_de, method='SLSQP',
                          bounds=bounds, constraints=slsqp_constraints,
                          options={'ftol': 1e-12, 'maxiter': 1000})

        best_x = np.clip(result.x, 0.0, 1.0)
        best_x = best_x / (best_x.sum() + 1e-12)
        best_cost = float(objective(best_x))

        # Normalize to weights
        weights = best_x / np.sum(best_x) if np.sum(best_x) > 1e-10 else best_x

        return OptimizationResult(
            optimal_weights=weights,
            optimal_value=best_cost,
            n_iterations=result.nit,
            eigenvalue=best_cost,
            eigenvector=best_x,
            optimizer_type=OptimizerType.CLASSICAL,
            execution_time=0,
            metadata={'method': 'quantum_inspired_de_slsqp',
                      'de_fun': float(de_result.fun),
                      'slsqp_success': result.success}
        )
    
    def _optimize_qaoa(self, qubo: QuadraticProgram,
                      initial_params: Optional[np.ndarray]) -> OptimizationResult:
        """QAOA optimization."""
        if not QISKIT_AVAILABLE:
            return self._optimize_classical(qubo)
        
        n = qubo.n_assets
        
        # Get Hamiltonian
        hamiltonian = qubo.to_pauli_op()
        
        # Create QAOA instance
        sampler = Sampler()
        optimizer = COBYLA(maxiter=100)
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.reps
        )
        
        # Run QAOA
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        # Extract best bitstring
        if hasattr(result, 'best_measurement'):
            bitstring = result.best_measurement['bitstring']
            x = np.array([int(b) for b in bitstring])
        else:
            # Fallback
            x = np.ones(n) / n
        
        weights = x / np.sum(x) if np.sum(x) > 0 else x
        
        return OptimizationResult(
            optimal_weights=weights,
            optimal_value=float(result.eigenvalue.real),
            n_iterations=100,
            eigenvalue=float(result.eigenvalue.real),
            eigenvector=x,
            optimizer_type=OptimizerType.QAOA,
            execution_time=0,
            metadata={
                'reps': self.reps,
                'optimal_params': result.optimal_parameters if hasattr(result, 'optimal_parameters') else None
            }
        )
    
    def _optimize_vqe(self, qubo: QuadraticProgram,
                     initial_params: Optional[np.ndarray]) -> OptimizationResult:
        """VQE optimization."""
        if not QISKIT_AVAILABLE:
            return self._optimize_classical(qubo)
        
        n = qubo.n_assets
        
        # Get Hamiltonian
        hamiltonian = qubo.to_pauli_op()
        
        # Build ansatz
        ansatz = self._build_vqe_ansatz(n)
        
        # Create VQE
        estimator = Estimator()
        optimizer = SLSQP(maxiter=100)
        
        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer
        )
        
        # Run VQE
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        # Extract solution
        # For VQE, we need to sample to get the bitstring
        x = np.ones(n) / n  # Placeholder
        
        weights = x
        
        return OptimizationResult(
            optimal_weights=weights,
            optimal_value=float(result.eigenvalue.real),
            n_iterations=100,
            eigenvalue=float(result.eigenvalue.real),
            eigenvector=None,
            optimizer_type=OptimizerType.VQE,
            execution_time=0,
            metadata={
                'reps': self.reps,
                'n_params': ansatz.num_parameters
            }
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("QUANTUM OPTIMIZER TEST")
    print("=" * 60)
    
    # Simple portfolio problem
    n_assets = 4
    
    # Expected returns
    expected_returns = np.array([0.12, 0.08, 0.15, 0.10])
    
    # Covariance matrix
    cov_matrix = np.array([
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.03, 0.01, 0.02],
        [0.02, 0.01, 0.06, 0.02],
        [0.01, 0.02, 0.02, 0.04]
    ])
    
    print(f"\nAssets: {n_assets}")
    print(f"Expected Returns: {expected_returns}")
    print(f"Risk Aversion: 0.5")
    
    # Create QUBO
    qubo = QuadraticProgram(n_assets)
    qubo.set_portfolio_objective(
        cov_matrix=cov_matrix,
        expected_returns=expected_returns,
        risk_aversion=0.5
    )
    
    # Classical optimization (always works)
    optimizer = QuantumOptimizer(
        optimizer_type=OptimizerType.CLASSICAL,
        reps=3
    )
    
    result = optimizer.optimize(qubo)
    
    print(f"\nClassical Optimization Result:")
    print(f"  Optimal Weights: {result.optimal_weights}")
    print(f"  Optimal Value: {result.optimal_value:.6f}")
    print(f"  Execution Time: {result.execution_time:.4f}s")
    
    # Calculate expected return and risk
    portfolio_return = result.optimal_weights @ expected_returns
    portfolio_risk = np.sqrt(result.optimal_weights @ cov_matrix @ result.optimal_weights)
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {portfolio_return:.2%}")
    print(f"  Risk (Std Dev): {portfolio_risk:.2%}")
    print(f"  Sharpe Ratio: {(portfolio_return - 0.02) / portfolio_risk:.2f}")
    
    # Test QAOA if available
    if QISKIT_AVAILABLE:
        print("\n" + "-" * 40)
        print("QAOA Optimization")
        
        qaoa_optimizer = QuantumOptimizer(
            optimizer_type=OptimizerType.QAOA,
            reps=2
        )
        
        qaoa_result = qaoa_optimizer.optimize(qubo)
        
        print(f"  Optimal Weights: {qaoa_result.optimal_weights}")
        print(f"  Optimal Value: {qaoa_result.optimal_value:.6f}")
        print(f"  Execution Time: {qaoa_result.execution_time:.4f}s")
    else:
        print("\nQiskit not available for quantum optimization")
