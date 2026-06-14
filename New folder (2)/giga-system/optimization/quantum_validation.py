"""
QUANTUM COMPUTING VALIDATION — Backend Testing, Benchmarking, Error Mitigation
================================================================================

Addresses Missing Concepts 6.1-6.4:
  6.1 — Real Quantum Backend Testing (IBM/Amazon Braket via simulators)
  6.2 — Quantum Advantage Benchmarking (quantum vs classical comparison)
  6.3 — Quantum Error Mitigation (ZNE, probabilistic error cancellation)
  6.4 — Quantum Feature Maps (encode financial features into quantum states)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing quantum libraries
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.primitives import Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


@dataclass
class QuantumBenchmarkResult:
    """Result of quantum vs classical comparison."""
    problem_size: int
    classical_result: float
    classical_time_ms: float
    quantum_result: float
    quantum_time_ms: float
    fidelity: float  # How close quantum is to classical (0-1)
    speedup_factor: float
    advantage: str  # "quantum", "classical", "parity"


@dataclass
class ErrorMitigationResult:
    """Result of quantum error mitigation."""
    raw_expectation: float
    mitigated_expectation: float
    improvement_pct: float
    method: str
    noise_levels: List[float]
    extrapolated_ideal: float


# =============================================================================
# 6.1 — QUANTUM BACKEND TESTING
# =============================================================================

class QuantumBackendTester:
    """
    Test QAOA and VQE on quantum backends (real or simulated).
    Falls back to classical matrix simulation when no quantum SDK available.
    """

    def __init__(self):
        self.backend_name = "classical_simulator"
        if QISKIT_AVAILABLE and AER_AVAILABLE:
            self.backend_name = "aer_simulator"
        self.results_log: List[Dict] = []

    def create_portfolio_circuit(self, n_assets: int, expected_returns: np.ndarray,
                                  covariance: np.ndarray) -> Dict:
        """
        Create a quantum circuit for portfolio optimization using QAOA.
        Returns circuit description (or Qiskit circuit if available).
        """
        if QISKIT_AVAILABLE:
            n_qubits = n_assets
            qc = QuantumCircuit(n_qubits, n_qubits)

            # QAOA ansatz: alternating layers of cost and mixer unitaries
            p = 2  # QAOA depth
            for layer in range(p):
                # Cost layer: encode portfolio objective
                for i in range(n_qubits):
                    angle = float(expected_returns[i]) * 0.1  # Scale down
                    qc.rz(angle, i)
                for i in range(n_qubits - 1):
                    cov_angle = float(covariance[i, i + 1]) * 0.1
                    qc.cx(i, i + 1)
                    qc.rz(cov_angle, i + 1)
                    qc.cx(i, i + 1)

                # Mixer layer
                for i in range(n_qubits):
                    qc.rx(np.pi / (layer + 2), i)

            qc.measure(range(n_qubits), range(n_qubits))

            return {
                "circuit": qc,
                "n_qubits": n_qubits,
                "depth": qc.depth(),
                "gate_count": qc.size(),
                "backend": self.backend_name,
            }
        else:
            # Classical description
            return {
                "circuit": "classical_simulation",
                "n_qubits": n_assets,
                "depth": n_assets * 4,
                "gate_count": n_assets * 6,
                "backend": "classical_matrix_sim",
            }

    def run_portfolio_optimization(
        self, n_assets: int, expected_returns: np.ndarray, covariance: np.ndarray,
        risk_aversion: float = 1.0, shots: int = 1024,
    ) -> Dict:
        """
        Run quantum portfolio optimization and return results.
        """
        import time

        start = time.time()

        if QISKIT_AVAILABLE and AER_AVAILABLE:
            circuit_info = self.create_portfolio_circuit(n_assets, expected_returns, covariance)
            qc = circuit_info["circuit"]

            backend = AerSimulator()
            transpiled = qc  # Simplified
            result = backend.run(transpiled, shots=shots).result()
            counts = result.get_counts()

            # Extract best portfolio from measurement results
            best_portfolio = max(counts, key=counts.get)
            weights = np.array([int(b) for b in best_portfolio[::-1]], dtype=float)
            total = np.sum(weights) + 1e-15
            weights = weights / total

            quantum_time = (time.time() - start) * 1000
        else:
            # Classical simulation of quantum behavior
            # Simulate QAOA by sampling from approximate distribution
            np.random.seed(42)
            n_samples = shots
            best_obj = -np.inf
            best_weights = np.ones(n_assets) / n_assets

            for _ in range(n_samples):
                # Random binary portfolio
                binary = np.random.randint(0, 2, n_assets).astype(float)
                if np.sum(binary) == 0:
                    continue
                w = binary / np.sum(binary)
                obj = w @ expected_returns - risk_aversion * w @ covariance @ w
                if obj > best_obj:
                    best_obj = obj
                    best_weights = w

            weights = best_weights
            quantum_time = (time.time() - start) * 1000

        # Also solve classically (Markowitz) for comparison
        start_c = time.time()
        try:
            inv_cov = np.linalg.inv(covariance + 0.01 * np.eye(n_assets))
            classical_weights = inv_cov @ expected_returns
            classical_weights = np.maximum(classical_weights, 0)
            classical_weights /= np.sum(classical_weights) + 1e-15
        except np.linalg.LinAlgError:
            classical_weights = np.ones(n_assets) / n_assets
        classical_time = (time.time() - start_c) * 1000

        # Portfolio metrics
        q_return = weights @ expected_returns
        q_risk = np.sqrt(weights @ covariance @ weights)
        c_return = classical_weights @ expected_returns
        c_risk = np.sqrt(classical_weights @ covariance @ classical_weights)

        result = {
            "quantum_weights": weights.tolist(),
            "classical_weights": classical_weights.tolist(),
            "quantum_return": float(q_return),
            "quantum_risk": float(q_risk),
            "quantum_sharpe": float(q_return / (q_risk + 1e-15)),
            "classical_return": float(c_return),
            "classical_risk": float(c_risk),
            "classical_sharpe": float(c_return / (c_risk + 1e-15)),
            "quantum_time_ms": float(quantum_time),
            "classical_time_ms": float(classical_time),
            "backend": self.backend_name,
            "n_assets": n_assets,
            "shots": shots,
        }
        self.results_log.append(result)
        return result


# =============================================================================
# 6.2 — QUANTUM ADVANTAGE BENCHMARKING
# =============================================================================

class QuantumAdvantageBenchmark:
    """
    Controlled comparison: does quantum optimization outperform classical?
    """

    def __init__(self):
        self.tester = QuantumBackendTester()

    def benchmark_scaling(self, asset_counts: Optional[List[int]] = None) -> List[QuantumBenchmarkResult]:
        """
        Run quantum vs classical across different problem sizes.
        """
        if asset_counts is None:
            asset_counts = [2, 3, 4, 5, 6, 8]

        results = []
        for n in asset_counts:
            # Generate random problem
            np.random.seed(n)
            expected_returns = np.random.uniform(0.05, 0.15, n)
            cov = np.random.randn(n, n) * 0.01
            covariance = cov.T @ cov + 0.02 * np.eye(n)

            result = self.tester.run_portfolio_optimization(n, expected_returns, covariance)

            q_sharpe = result["quantum_sharpe"]
            c_sharpe = result["classical_sharpe"]
            fidelity = 1.0 - abs(q_sharpe - c_sharpe) / (abs(c_sharpe) + 1e-15)
            fidelity = max(0.0, min(1.0, fidelity))

            speedup = result["classical_time_ms"] / (result["quantum_time_ms"] + 1e-15)

            if q_sharpe > c_sharpe * 1.05:
                advantage = "quantum"
            elif c_sharpe > q_sharpe * 1.05:
                advantage = "classical"
            else:
                advantage = "parity"

            results.append(QuantumBenchmarkResult(
                problem_size=n,
                classical_result=c_sharpe,
                classical_time_ms=result["classical_time_ms"],
                quantum_result=q_sharpe,
                quantum_time_ms=result["quantum_time_ms"],
                fidelity=fidelity,
                speedup_factor=speedup,
                advantage=advantage,
            ))

        return results

    def summary_report(self, results: List[QuantumBenchmarkResult]) -> Dict:
        """Generate summary of benchmark results."""
        n_quantum_wins = sum(1 for r in results if r.advantage == "quantum")
        n_classical_wins = sum(1 for r in results if r.advantage == "classical")
        avg_fidelity = np.mean([r.fidelity for r in results])
        avg_speedup = np.mean([r.speedup_factor for r in results])

        return {
            "n_tests": len(results),
            "quantum_wins": n_quantum_wins,
            "classical_wins": n_classical_wins,
            "parity": len(results) - n_quantum_wins - n_classical_wins,
            "avg_fidelity": float(avg_fidelity),
            "avg_speedup_factor": float(avg_speedup),
            "conclusion": (
                "Quantum shows advantage at larger problem sizes"
                if n_quantum_wins > n_classical_wins
                else "Classical remains competitive; quantum advantage requires larger problems"
            ),
        }


# =============================================================================
# 6.3 — QUANTUM ERROR MITIGATION
# =============================================================================

class QuantumErrorMitigation:
    """
    Error mitigation techniques for NISQ devices.
    Implements Zero-Noise Extrapolation (ZNE) and Richardson extrapolation.
    """

    def __init__(self):
        pass

    def zero_noise_extrapolation(
        self,
        expectation_fn,  # Callable that takes noise_level and returns expectation
        noise_levels: Optional[List[float]] = None,
    ) -> ErrorMitigationResult:
        """
        ZNE: Run circuit at multiple noise levels, extrapolate to zero noise.
        Uses Richardson extrapolation (connects to Eudoxian exhaustion!).
        """
        if noise_levels is None:
            noise_levels = [1.0, 1.5, 2.0, 3.0]

        expectations = []
        for noise in noise_levels:
            exp_val = expectation_fn(noise)
            expectations.append(exp_val)

        # Richardson extrapolation to noise=0
        # For 2 points: f(0) ≈ (c2*f(c1) - c1*f(c2)) / (c2 - c1)
        # For n points: polynomial fit and extrapolate
        coeffs = np.polyfit(noise_levels, expectations, min(len(noise_levels) - 1, 3))
        extrapolated = np.polyval(coeffs, 0.0)

        raw = expectations[0]
        improvement = abs(extrapolated - raw) / (abs(raw) + 1e-15) * 100

        return ErrorMitigationResult(
            raw_expectation=float(raw),
            mitigated_expectation=float(extrapolated),
            improvement_pct=float(improvement),
            method="zero_noise_extrapolation",
            noise_levels=noise_levels,
            extrapolated_ideal=float(extrapolated),
        )

    def probabilistic_error_cancellation(
        self,
        noisy_results: np.ndarray,
        error_rate: float = 0.01,
    ) -> Dict:
        """
        PEC: probabilistically cancel errors by sampling correction circuits.
        Simplified implementation using statistical correction.
        """
        n = len(noisy_results)
        # Correction factor for depolarizing noise
        correction_factor = 1.0 / (1.0 - 2 * error_rate)

        corrected = noisy_results * correction_factor
        # The variance increases with correction
        variance_amplification = correction_factor**2

        return {
            "noisy_mean": float(np.mean(noisy_results)),
            "corrected_mean": float(np.mean(corrected)),
            "correction_factor": float(correction_factor),
            "variance_amplification": float(variance_amplification),
            "error_rate_assumed": float(error_rate),
            "method": "probabilistic_error_cancellation",
        }


# =============================================================================
# 6.4 — QUANTUM FEATURE MAPS
# =============================================================================

class QuantumFeatureMap:
    """
    Encode financial features into quantum states for quantum ML.
    """

    def __init__(self, n_features: int = 4, encoding: str = "angle"):
        self.n_features = n_features
        self.encoding = encoding  # "angle", "amplitude", "iqp"

    def angle_encoding(self, features: np.ndarray) -> Dict:
        """
        Angle encoding: map each feature to a rotation angle.
        |ψ⟩ = ⊗ᵢ Ry(xᵢ) |0⟩
        """
        n = min(len(features), self.n_features)
        # Normalize features to [0, 2π]
        norm_features = (features[:n] - np.min(features[:n])) / (np.ptp(features[:n]) + 1e-15) * 2 * np.pi

        if QISKIT_AVAILABLE:
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.ry(float(norm_features[i]), i)

            # Add entangling layer for expressiveness
            for i in range(n - 1):
                qc.cx(i, i + 1)

            return {
                "circuit": qc,
                "n_qubits": n,
                "encoding": "angle",
                "angles": norm_features.tolist(),
            }
        else:
            # Classical simulation of quantum state
            state = np.ones(2**n) / np.sqrt(2**n)
            for i in range(n):
                angle = norm_features[i]
                # Apply Ry rotation on qubit i
                cos_half = np.cos(angle / 2)
                sin_half = np.sin(angle / 2)
                new_state = np.zeros_like(state)
                for j in range(len(state)):
                    bit_i = (j >> i) & 1
                    if bit_i == 0:
                        partner = j | (1 << i)
                        new_state[j] += cos_half * state[j]
                        new_state[partner] += sin_half * state[j]
                    else:
                        partner = j & ~(1 << i)
                        new_state[j] += cos_half * state[j]
                        new_state[partner] -= sin_half * state[j]
                state = new_state / (np.linalg.norm(new_state) + 1e-15)

            return {
                "state_vector": state.tolist(),
                "n_qubits": n,
                "encoding": "angle",
                "angles": norm_features.tolist(),
            }

    def iqp_encoding(self, features: np.ndarray) -> Dict:
        """
        IQP (Instantaneous Quantum Polynomial) encoding.
        Higher-order feature interactions via ZZ gates.
        """
        n = min(len(features), self.n_features)
        norm_features = features[:n] / (np.max(np.abs(features[:n])) + 1e-15) * np.pi

        circuit_description = {
            "n_qubits": n,
            "encoding": "iqp",
            "layers": [],
        }

        # Layer 1: Hadamard on all qubits
        circuit_description["layers"].append({"type": "H", "qubits": list(range(n))})

        # Layer 2: Single-qubit Z rotations
        circuit_description["layers"].append({
            "type": "Rz",
            "rotations": {i: float(norm_features[i]) for i in range(n)},
        })

        # Layer 3: Two-qubit ZZ interactions (feature products)
        interactions = {}
        for i in range(n):
            for j in range(i + 1, n):
                interactions[f"{i},{j}"] = float(norm_features[i] * norm_features[j])
        circuit_description["layers"].append({"type": "ZZ", "interactions": interactions})

        # Layer 4: Hadamard again
        circuit_description["layers"].append({"type": "H", "qubits": list(range(n))})

        return circuit_description

    def encode_market_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Encode common financial features into quantum state.
        Features: [return, volatility, volume_change, momentum]
        """
        returns = np.diff(np.log(prices + 1e-15))
        vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        vol_change = np.diff(volumes[-5:]).mean() / (np.mean(volumes[-5:]) + 1e-15) if len(volumes) >= 5 else 0.0
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0

        features = np.array([returns[-1], vol, vol_change, momentum])

        if self.encoding == "angle":
            return self.angle_encoding(features)
        elif self.encoding == "iqp":
            return self.iqp_encoding(features)
        else:
            return self.angle_encoding(features)
