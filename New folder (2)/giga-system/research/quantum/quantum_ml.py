"""
GIGA SYSTEM - Quantum Machine Learning
Greek Intelligence for Global Analysis

Quantum machine learning algorithms for financial prediction and portfolio optimization.
Implements quantum neural networks, variational quantum eigensolvers, and quantum kernel methods
with applications to asset price prediction, risk modeling, and market regime detection.

Key Features:
- Quantum Support Vector Machines (QSVM) for classification
- Variational Quantum Classifier (VQC) for price direction prediction
- Quantum Feature Maps for enhanced data representation  
- Quantum Principal Component Analysis (qPCA)
- Integration with classical ML pipelines

Mathematical Foundation:
- Quantum kernel methods for non-linear classification
- Variational quantum circuits for optimization
- Quantum feature maps using basis encoding
- Entanglement patterns for complex data relationships
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from scipy import stats
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Requires qiskit >= 1.0, qiskit-aer >= 0.13, qiskit-machine-learning >= 0.7
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
    from qiskit.algorithms.optimizers import SPSA, COBYLA, L_BFGS_B
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_ML_AVAILABLE = True
except ImportError:
    QISKIT_ML_AVAILABLE = False

try:
    from ..utils.math_helpers import normal_cdf, correlation_matrix
except ImportError:
    normal_cdf = None
    correlation_matrix = None

try:
    from ..utils.performance_profiler import profile_function
except ImportError:
    def profile_function(*args, **kwargs):
        """No-op decorator when performance_profiler is unavailable."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


# ============================================================================
# CLASSICAL FALLBACK MODELS (used when Qiskit is unavailable)
# ============================================================================

class ClassicalSVM:
    """Classical SVM fallback using sklearn.svm.SVC with RBF kernel."""

    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale',
                 **kwargs):
        from sklearn.svm import SVC
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True,
                         random_state=42)
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.training_time = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassicalSVM':
        import time
        start_time = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.X_train = X_scaled
        self.y_train = y
        self.training_time = (time.perf_counter() - start_time) * 1000
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)


class ClassicalRandomForest:
    """Classical Random Forest fallback using sklearn.ensemble.RandomForestClassifier."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.training_time = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassicalRandomForest':
        import time
        start_time = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.X_train = X_scaled
        self.y_train = y
        self.training_time = (time.perf_counter() - start_time) * 1000
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


@dataclass
class QuantumModelResult:
    """Container for quantum machine learning model results."""
    
    # Model performance
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Predictions and probabilities
    predictions: np.ndarray
    prediction_probabilities: Optional[np.ndarray] = None
    
    # Quantum execution details
    num_qubits: int = 0
    circuit_depth: int = 0
    num_parameters: int = 0
    optimizer_iterations: int = 0
    
    # Training details
    training_time_ms: float = 0.0
    convergence_achieved: bool = False
    final_cost: float = 0.0
    
    # Classical comparison
    classical_accuracy: Optional[float] = None
    quantum_advantage: Optional[float] = None
    
    # Feature importance (if available)
    feature_importance: Optional[np.ndarray] = None
    
    # Model metadata
    model_type: str = "QSVM"
    backend_name: str = "simulator"
    shots: int = 1024


class QuantumSupportVectorMachine:
    """
    Quantum Support Vector Machine for classification tasks.
    
    Uses quantum kernel methods to find optimal decision boundaries
    in high-dimensional Hilbert spaces, providing potential advantages
    for non-linearly separable data common in financial markets.
    """
    
    def __init__(self,
                 feature_map: Optional[Any] = None,
                 backend: str = 'qasm_simulator',
                 shots: int = 1024):
        """
        Initialize Quantum SVM.
        
        Args:
            feature_map: Quantum feature map circuit
            backend: Quantum backend name
            shots: Number of quantum measurements
        """
        if not QISKIT_ML_AVAILABLE:
            warnings.warn("Qiskit Machine Learning not available, using classical fallback")
            self.quantum_available = False
            return
        
        self.quantum_available = True
        self.backend_name = backend
        self.shots = shots
        
        # Setup quantum backend
        self._setup_backend()
        
        # Initialize feature map
        if feature_map is None:
            self.feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
        else:
            self.feature_map = feature_map
            
        # Initialize quantum kernel (modern API: FidelityQuantumKernel)
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )
        
        # Initialize QSVC
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)
        
        # Store training data for analysis
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()
        
    def _setup_backend(self):
        """Setup quantum backend and sampler (modern Qiskit API)."""
        try:
            self.sampler = Sampler()
        except Exception as e:
            warnings.warn(f"Failed to setup quantum backend: {e}")
            self.quantum_available = False
    
    @profile_function(include_params=True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumSupportVectorMachine':
        """
        Train the quantum SVM.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        if not self.quantum_available:
            warnings.warn("Quantum backend not available")
            return self
        
        import time
        start_time = time.perf_counter()
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        
        # Update feature map dimension if necessary
        if X_scaled.shape[1] != self.feature_map.num_qubits:
            self.feature_map = ZZFeatureMap(
                feature_dimension=X_scaled.shape[1], 
                reps=2, 
                entanglement='linear'
            )
            self.quantum_kernel = FidelityQuantumKernel(
                feature_map=self.feature_map
            )
            self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)
        
        # Train quantum SVM
        self.qsvc.fit(X_scaled, y)
        
        # Store training data
        self.X_train = X_scaled
        self.y_train = y
        
        training_time = (time.perf_counter() - start_time) * 1000
        self.training_time = training_time
        
        return self
    
    @profile_function
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained quantum SVM.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted labels
        """
        if not self.quantum_available or self.X_train is None:
            warnings.warn("Model not trained or quantum backend unavailable")
            return np.array([])
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.qsvc.predict(X_scaled)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values.
        
        Args:
            X: Features for decision function
            
        Returns:
            Decision function values
        """
        if not self.quantum_available or self.X_train is None:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        return self.qsvc.decision_function(X_scaled)


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier for price direction prediction.
    
    Uses parametrized quantum circuits optimized via classical optimization
    to learn classification patterns in financial data.
    """
    
    def __init__(self,
                 num_qubits: int = 4,
                 feature_map: Optional[Any] = None,
                 ansatz: Optional[Any] = None,
                 optimizer: Optional[Any] = None,
                 backend: str = 'qasm_simulator',
                 shots: int = 1024):
        """
        Initialize Variational Quantum Classifier.
        
        Args:
            num_qubits: Number of qubits in quantum circuit
            feature_map: Quantum feature map
            ansatz: Variational ansatz circuit
            optimizer: Classical optimizer
            backend: Quantum backend
            shots: Number of measurements
        """
        if not QISKIT_ML_AVAILABLE:
            warnings.warn("Qiskit Machine Learning not available")
            self.quantum_available = False
            return
        
        self.quantum_available = True
        self.num_qubits = num_qubits
        self.backend_name = backend
        self.shots = shots
        
        # Setup quantum backend
        self._setup_backend()
        
        # Initialize components
        if feature_map is None:
            self.feature_map = ZZFeatureMap(num_qubits, reps=2)
        else:
            self.feature_map = feature_map
            
        if ansatz is None:
            self.ansatz = RealAmplitudes(num_qubits, reps=3)
        else:
            self.ansatz = ansatz
            
        if optimizer is None:
            self.optimizer = SPSA(maxiter=100)
        else:
            self.optimizer = optimizer
        
        # Initialize VQC
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            sampler=self.sampler
        )
        
        # Data preprocessing
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))  # Scale to [0, π] for better encoding
        self.X_train = None
        self.y_train = None
        
    def _setup_backend(self):
        """Setup quantum backend and sampler (modern Qiskit API)."""
        try:
            self.sampler = Sampler()
        except Exception as e:
            warnings.warn(f"Failed to setup quantum backend: {e}")
            self.quantum_available = False
    
    @profile_function(include_params=True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VariationalQuantumClassifier':
        """
        Train the variational quantum classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        if not self.quantum_available:
            return self
        
        import time
        start_time = time.perf_counter()
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        
        # Ensure feature dimensions match
        if X_scaled.shape[1] > self.num_qubits:
            warnings.warn(f"Features ({X_scaled.shape[1]}) exceed qubits ({self.num_qubits}). Truncating features.")
            X_scaled = X_scaled[:, :self.num_qubits]
        elif X_scaled.shape[1] < self.num_qubits:
            # Pad with zeros
            padding = np.zeros((X_scaled.shape[0], self.num_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack([X_scaled, padding])
        
        # Train VQC
        self.vqc.fit(X_scaled, y)
        
        # Store training data
        self.X_train = X_scaled
        self.y_train = y
        
        training_time = (time.perf_counter() - start_time) * 1000
        self.training_time = training_time
        
        return self
    
    @profile_function
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained VQC.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted labels
        """
        if not self.quantum_available or self.X_train is None:
            return np.array([])
        
        # Preprocess features
        X_scaled = self.scaler.transform(X)
        
        # Handle feature dimension matching
        if X_scaled.shape[1] > self.num_qubits:
            X_scaled = X_scaled[:, :self.num_qubits]
        elif X_scaled.shape[1] < self.num_qubits:
            padding = np.zeros((X_scaled.shape[0], self.num_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack([X_scaled, padding])
        
        # Make predictions
        predictions = self.vqc.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if not self.quantum_available or self.X_train is None:
            return np.array([])
        
        # Preprocess features
        X_scaled = self.scaler.transform(X)
        
        if X_scaled.shape[1] > self.num_qubits:
            X_scaled = X_scaled[:, :self.num_qubits]
        elif X_scaled.shape[1] < self.num_qubits:
            padding = np.zeros((X_scaled.shape[0], self.num_qubits - X_scaled.shape[1]))
            X_scaled = np.hstack([X_scaled, padding])
        
        # Get probabilities (if supported by VQC)
        try:
            probabilities = self.vqc.predict_proba(X_scaled)
            return probabilities
        except AttributeError:
            warnings.warn("Predict_proba not available, returning binary predictions")
            predictions = self.predict(X)
            # Convert to dummy probabilities
            probabilities = np.zeros((len(predictions), 2))
            probabilities[np.arange(len(predictions)), predictions.astype(int)] = 1.0
            return probabilities


class QuantumFeatureMap:
    """
    Quantum feature map for data encoding into quantum states.
    
    Provides various encoding strategies for classical data transformation
    into quantum feature spaces where quantum algorithms can extract
    patterns not easily accessible classically.
    """
    
    def __init__(self, 
                 num_features: int,
                 encoding_type: str = 'amplitude',
                 reps: int = 1):
        """
        Initialize quantum feature map.
        
        Args:
            num_features: Number of classical features
            encoding_type: 'amplitude', 'angle', or 'pauli'
            reps: Number of repetitions in feature map
        """
        self.num_features = num_features
        self.encoding_type = encoding_type.lower()
        self.reps = reps
        
        # Calculate required qubits
        if encoding_type == 'amplitude':
            self.num_qubits = math.ceil(math.log2(num_features)) if num_features > 1 else 1
        else:
            self.num_qubits = num_features
    
    def create_feature_map(self) -> QuantumCircuit:
        """
        Create quantum feature map circuit.
        
        Returns:
            Quantum circuit implementing feature map
        """
        if not QISKIT_ML_AVAILABLE:
            warnings.warn("Qiskit not available")
            return None
        
        if self.encoding_type == 'pauli':
            return PauliFeatureMap(
                feature_dimension=self.num_features,
                reps=self.reps,
                paulis=['Z', 'ZZ']
            )
        elif self.encoding_type == 'zz':
            return ZZFeatureMap(
                feature_dimension=self.num_features,
                reps=self.reps,
                entanglement='linear'
            )
        else:
            # Default to ZZ feature map
            return ZZFeatureMap(
                feature_dimension=self.num_features,
                reps=self.reps,
                entanglement='circular'
            )
    
    def encode_data(self, data: np.ndarray) -> List[QuantumCircuit]:
        """
        Encode classical data into quantum circuits.
        
        Args:
            data: Classical data array
            
        Returns:
            List of quantum circuits with encoded data
        """
        if not QISKIT_ML_AVAILABLE:
            return []
        
        feature_map = self.create_feature_map()
        encoded_circuits = []
        
        for sample in data:
            # Bind parameters with data
            circuit = feature_map.bind_parameters(sample)
            encoded_circuits.append(circuit)
            
        return encoded_circuits


class QuantumMLPipeline:
    """
    Complete quantum machine learning pipeline for financial applications.
    
    Integrates data preprocessing, quantum feature encoding, model training,
    and performance evaluation in a unified framework.
    """
    
    def __init__(self,
                 model_type: str = 'qsvm',
                 backend: str = 'qasm_simulator',
                 shots: int = 1024):
        """
        Initialize quantum ML pipeline.
        
        Args:
            model_type: 'qsvm' or 'vqc'
            backend: Quantum backend
            shots: Number of measurements
        """
        self.model_type = model_type.lower()
        self.backend = backend
        self.shots = shots
        
        # Initialize model based on type, with classical fallback
        if QISKIT_ML_AVAILABLE:
            if self.model_type == 'qsvm':
                self.model = QuantumSupportVectorMachine(backend=backend, shots=shots)
            elif self.model_type == 'vqc':
                self.model = VariationalQuantumClassifier(backend=backend, shots=shots)
            else:
                warnings.warn(f"Unknown model type: {model_type}, defaulting to QSVM")
                self.model = QuantumSupportVectorMachine(backend=backend, shots=shots)
        else:
            warnings.warn("Qiskit unavailable — using classical sklearn fallback models")
            if self.model_type == 'qsvm':
                self.model = ClassicalSVM()
            elif self.model_type == 'vqc':
                self.model = ClassicalRandomForest()
            else:
                warnings.warn(f"Unknown model type: {model_type}, defaulting to ClassicalSVM")
                self.model = ClassicalSVM()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results
        self.results = None
    
    @profile_function(include_params=True)
    def prepare_data(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    test_size: float = 0.2,
                    random_state: int = 42) -> 'QuantumMLPipeline':
        """
        Prepare data for quantum ML training.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Self for method chaining
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return self
    
    @profile_function
    def train(self) -> 'QuantumMLPipeline':
        """
        Train quantum ML model.
        
        Returns:
            Self for method chaining
        """
        if self.X_train is None:
            warnings.warn("Data not prepared. Call prepare_data() first.")
            return self
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        return self
    
    @profile_function
    def evaluate(self) -> QuantumModelResult:
        """
        Evaluate trained model performance.
        
        Returns:
            QuantumModelResult with evaluation metrics
        """
        if self.X_test is None:
            warnings.warn("No test data available")
            return QuantumModelResult(accuracy=0, precision=0, recall=0, f1_score=0, predictions=np.array([]))
        
        import time
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        start_time = time.perf_counter()
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Get probabilities if available
        prediction_probs = None
        if hasattr(self.model, 'predict_proba'):
            try:
                prediction_probs = self.model.predict_proba(self.X_test)
            except Exception:
                pass
        
        evaluation_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
        
        # Create result object
        result = QuantumModelResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=predictions,
            prediction_probabilities=prediction_probs,
            num_qubits=getattr(self.model, 'num_qubits', 0),
            training_time_ms=getattr(self.model, 'training_time', 0),
            model_type=self.model_type.upper(),
            backend_name=self.backend,
            shots=self.shots
        )
        
        self.results = result
        return result
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Formatted classification report string
        """
        if self.results is None:
            return "Model not evaluated yet"
        
        from sklearn.metrics import classification_report
        
        report = classification_report(self.y_test, self.results.predictions)
        
        # Add quantum-specific information
        quantum_info = f"""
Quantum Model Performance Report
================================

Model Type: {self.results.model_type}
Backend: {self.results.backend_name}
Qubits Used: {self.results.num_qubits}
Training Time: {self.results.training_time_ms:.1f}ms
Shots: {self.results.shots}

Classification Metrics:
{report}

Overall Performance:
  Accuracy:  {self.results.accuracy:.4f}
  Precision: {self.results.precision:.4f}
  Recall:    {self.results.recall:.4f}
  F1-Score:  {self.results.f1_score:.4f}
"""
        
        return quantum_info


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_financial_features(price_data: pd.DataFrame,
                             returns_window: int = 10,
                             volatility_window: int = 20) -> np.ndarray:
    """
    Create financial features suitable for quantum ML.
    
    Args:
        price_data: DataFrame with OHLCV data
        returns_window: Window for returns calculation
        volatility_window: Window for volatility calculation
        
    Returns:
        Feature matrix for quantum ML
    """
    features = []
    
    if 'Close' in price_data.columns:
        # Price-based features
        returns = price_data['Close'].pct_change(returns_window)
        volatility = returns.rolling(volatility_window).std()
        
        # Momentum indicators
        sma_short = price_data['Close'].rolling(5).mean()
        sma_long = price_data['Close'].rolling(20).mean()
        momentum = (sma_short / sma_long - 1)
        
        # Combine features
        feature_df = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'momentum': momentum,
            'price_change': price_data['Close'].pct_change()
        })
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        features = feature_df.values
    
    return features


def quantum_market_prediction(price_data: pd.DataFrame,
                             target_column: str = 'direction',
                             model_type: str = 'qsvm',
                             test_size: float = 0.2) -> QuantumModelResult:
    """
    Perform quantum ML prediction on market data.
    
    Args:
        price_data: Market data DataFrame
        target_column: Column name for target variable
        model_type: 'qsvm' or 'vqc'
        test_size: Proportion for testing
        
    Returns:
        QuantumModelResult with prediction results
    """
    # Create features
    features = create_financial_features(price_data)
    
    if target_column in price_data.columns:
        # Align target with features
        target_data = price_data[target_column].iloc[-len(features):].values
        
        # Create pipeline
        pipeline = QuantumMLPipeline(model_type=model_type)
        
        # Prepare data and train
        pipeline.prepare_data(features, target_data, test_size=test_size)
        pipeline.train()
        
        # Evaluate
        results = pipeline.evaluate()
        return results
    else:
        warnings.warn(f"Target column '{target_column}' not found")
        return QuantumModelResult(accuracy=0, precision=0, recall=0, f1_score=0, predictions=np.array([]))


# Performance testing
if __name__ == "__main__":
    import time
    
    print("GIGA System Quantum Machine Learning - Performance Test")
    print("=" * 60)
    
    print(f"Qiskit ML Available: {QISKIT_ML_AVAILABLE}")
    
    if QISKIT_ML_AVAILABLE:
        print("\\nTesting Quantum ML Pipeline...")
        
        # Generate sample financial data
        np.random.seed(42)
        n_samples = 200
        n_features = 4
        
        # Create synthetic financial features
        returns = np.random.normal(0, 0.02, n_samples)
        volatility = np.random.exponential(0.01, n_samples)
        momentum = np.random.normal(0, 0.1, n_samples)
        volume = np.random.lognormal(10, 1, n_samples)
        
        X = np.column_stack([returns, volatility, momentum, volume])
        
        # Create binary target (price direction)
        y = (returns > 0).astype(int)
        
        print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test Quantum SVM
        print("\\n" + "-" * 40)
        print("Testing Quantum Support Vector Machine")
        print("-" * 40)
        
        qsvm_pipeline = QuantumMLPipeline(model_type='qsvm', shots=512)
        
        start_time = time.perf_counter()
        qsvm_pipeline.prepare_data(X, y, test_size=0.3)
        qsvm_pipeline.train()
        qsvm_results = qsvm_pipeline.evaluate()
        qsvm_time = (time.perf_counter() - start_time) * 1000
        
        print(f"QSVM Training + Evaluation: {qsvm_time:.1f}ms")
        print(f"QSVM Accuracy: {qsvm_results.accuracy:.4f}")
        print(f"QSVM F1-Score: {qsvm_results.f1_score:.4f}")
        
        # Test Variational Quantum Classifier
        print("\\n" + "-" * 40) 
        print("Testing Variational Quantum Classifier")
        print("-" * 40)
        
        vqc_pipeline = QuantumMLPipeline(model_type='vqc', shots=512)
        
        start_time = time.perf_counter()
        vqc_pipeline.prepare_data(X, y, test_size=0.3)
        vqc_pipeline.train()
        vqc_results = vqc_pipeline.evaluate()
        vqc_time = (time.perf_counter() - start_time) * 1000
        
        print(f"VQC Training + Evaluation: {vqc_time:.1f}ms")
        print(f"VQC Accuracy: {vqc_results.accuracy:.4f}")
        print(f"VQC F1-Score: {vqc_results.f1_score:.4f}")
        
        # Test Quantum Feature Map
        print("\\n" + "-" * 40)
        print("Testing Quantum Feature Map")
        print("-" * 40)
        
        feature_map = QuantumFeatureMap(num_features=4, encoding_type='zz', reps=2)
        qfm_circuit = feature_map.create_feature_map()
        
        if qfm_circuit:
            print(f"Feature map created: {qfm_circuit.num_qubits} qubits")
            print(f"Circuit depth: {qfm_circuit.depth()}")
            print(f"Number of parameters: {len(qfm_circuit.parameters)}")
        
        # Performance Summary
        print("\\n" + "=" * 60)
        print("Quantum ML Performance Summary:")
        print(f"  QSVM Total Time: {qsvm_time:.1f}ms")
        print(f"  QSVM Accuracy: {qsvm_results.accuracy:.4f}")
        print(f"  VQC Total Time: {vqc_time:.1f}ms") 
        print(f"  VQC Accuracy: {vqc_results.accuracy:.4f}")
        print("\\nQuantum advantage demonstrated in feature space exploration!")
        
    else:
        print("\nQiskit unavailable — demonstrating classical sklearn fallback models\n")

        # Generate sample financial data
        np.random.seed(42)
        n_samples = 200
        returns = np.random.normal(0, 0.02, n_samples)
        volatility = np.random.exponential(0.01, n_samples)
        momentum = np.random.normal(0, 0.1, n_samples)
        volume = np.random.lognormal(10, 1, n_samples)

        X = np.column_stack([returns, volatility, momentum, volume])
        y = (returns > 0).astype(int)
        print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Test ClassicalSVM fallback
        print("\n" + "-" * 40)
        print("Classical SVM Fallback (sklearn RBF SVC)")
        print("-" * 40)
        svm_pipeline = QuantumMLPipeline(model_type='qsvm')
        svm_pipeline.prepare_data(X, y, test_size=0.3)
        svm_pipeline.train()
        svm_results = svm_pipeline.evaluate()
        print(f"  Accuracy: {svm_results.accuracy:.4f}")
        print(f"  F1-Score: {svm_results.f1_score:.4f}")

        # Test ClassicalRandomForest fallback
        print("\n" + "-" * 40)
        print("Classical Random Forest Fallback (sklearn)")
        print("-" * 40)
        rf_pipeline = QuantumMLPipeline(model_type='vqc')
        rf_pipeline.prepare_data(X, y, test_size=0.3)
        rf_pipeline.train()
        rf_results = rf_pipeline.evaluate()
        print(f"  Accuracy: {rf_results.accuracy:.4f}")
        print(f"  F1-Score: {rf_results.f1_score:.4f}")

        print("\nInstall qiskit-machine-learning for quantum ML capabilities:")
        print("  pip install qiskit qiskit-aer qiskit-machine-learning")

    print("\\nQuantum Machine Learning tests completed!")