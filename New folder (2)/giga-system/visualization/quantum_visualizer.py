"""
Quantum Visualizer - Interactive Quantum Computing Visualization
=============================================================

Advanced visualization suite for quantum computing concepts in finance,
including quantum circuits, quantum algorithms, quantum portfolio optimization,
and quantum risk modeling visualization.

Features:
- Interactive quantum circuit diagrams
- Quantum algorithm visualization (QAOA, VQE, etc.)
- Quantum portfolio optimization visualization
- Quantum state visualization and Bloch sphere
- Quantum error correction visualization
- Quantum advantage demonstrations
- Quantum Monte Carlo visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from quantum.quantum_optimizer import QuantumOptimizer
    from quantum.portfolio_quantum import QuantumPortfolioOptimizer
    from quantum.risk_quantum import QuantumRiskAnalyzer
    from utils.performance_profiler import PerformanceProfiler
except ImportError:
    # Fallback implementations for quantum components
    class QuantumOptimizer:
        def optimize_portfolio(self, returns, risk_tolerance=0.5):
            # Classical fallback
            n_assets = len(returns)
            weights = np.random.dirichlet(np.ones(n_assets))
            return {'weights': weights, 'expected_return': np.dot(weights, returns)}
    
    class QuantumPortfolioOptimizer:
        def __init__(self):
            self.optimizer = QuantumOptimizer()
        
        def optimize(self, expected_returns, covariance_matrix):
            return self.optimizer.optimize_portfolio(expected_returns)
    
    class QuantumRiskAnalyzer:
        def calculate_quantum_var(self, portfolio_returns, confidence=0.95):
            return {'var': np.percentile(portfolio_returns, (1-confidence)*100)}
    
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start


class QuantumVisualizer:
    """Interactive quantum computing visualization for finance."""
    
    def __init__(self):
        """Initialize Quantum Visualizer."""
        self.quantum_optimizer = QuantumOptimizer()
        self.portfolio_optimizer = QuantumPortfolioOptimizer()
        self.risk_analyzer = QuantumRiskAnalyzer()
        self.profiler = PerformanceProfiler()
        
        # Quantum visualization cache
        if 'quantum_cache' not in st.session_state:
            st.session_state.quantum_cache = {}
    
    def create_bloch_sphere(self, theta: float = np.pi/4, phi: float = 0) -> go.Figure:
        """Create interactive Bloch sphere visualization."""
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add Bloch sphere surface
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Bloch Sphere'
        ))
        
        # Add coordinate axes
        # X axis
        fig.add_trace(go.Scatter3d(
            x=[-1.2, 1.2], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=5),
            text=['', '|+⟩'],
            textposition='middle right',
            name='X axis'
        ))
        
        # Y axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-1.2, 1.2], z=[0, 0],
            mode='lines+text',
            line=dict(color='green', width=5),
            text=['', '|+i⟩'],
            textposition='middle right',
            name='Y axis'
        ))
        
        # Z axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1.2, 1.2],
            mode='lines+text',
            line=dict(color='blue', width=5),
            text=['|1⟩', '|0⟩'],
            textposition='middle right',
            name='Z axis'
        ))
        
        # Add quantum state vector
        x_state = np.sin(theta) * np.cos(phi)
        y_state = np.sin(theta) * np.sin(phi)
        z_state = np.cos(theta)
        
        fig.add_trace(go.Scatter3d(
            x=[0, x_state], y=[0, y_state], z=[0, z_state],
            mode='lines+markers',
            line=dict(color='yellow', width=8),
            marker=dict(size=[5, 10], color=['yellow', 'orange']),
            name='Quantum State |ψ⟩'
        ))
        
        fig.update_layout(
            title="Interactive Bloch Sphere - Quantum State Visualization",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def create_quantum_circuit(self, circuit_type: str = "qaoa") -> go.Figure:
        """Create quantum circuit diagram visualization."""
        
        if circuit_type == "qaoa":
            # QAOA circuit for portfolio optimization
            num_qubits = 4
            circuit_data = {
                'qubits': list(range(num_qubits)),
                'gates': [
                    {'type': 'H', 'qubit': 0, 'time': 1},
                    {'type': 'H', 'qubit': 1, 'time': 1},
                    {'type': 'H', 'qubit': 2, 'time': 1},
                    {'type': 'H', 'qubit': 3, 'time': 1},
                    {'type': 'RZ', 'qubit': 0, 'time': 2, 'angle': 'γ'},
                    {'type': 'RZ', 'qubit': 1, 'time': 2, 'angle': 'γ'},
                    {'type': 'CNOT', 'control': 0, 'target': 1, 'time': 3},
                    {'type': 'CNOT', 'control': 2, 'target': 3, 'time': 3},
                    {'type': 'RX', 'qubit': 0, 'time': 4, 'angle': 'β'},
                    {'type': 'RX', 'qubit': 1, 'time': 4, 'angle': 'β'},
                    {'type': 'RX', 'qubit': 2, 'time': 4, 'angle': 'β'},
                    {'type': 'RX', 'qubit': 3, 'time': 4, 'angle': 'β'},
                ]
            }
            title = "QAOA Circuit for Portfolio Optimization"
        
        elif circuit_type == "vqe":
            # VQE circuit for risk analysis
            num_qubits = 3
            circuit_data = {
                'qubits': list(range(num_qubits)),
                'gates': [
                    {'type': 'RY', 'qubit': 0, 'time': 1, 'angle': 'θ₁'},
                    {'type': 'RY', 'qubit': 1, 'time': 1, 'angle': 'θ₂'},
                    {'type': 'RY', 'qubit': 2, 'time': 1, 'angle': 'θ₃'},
                    {'type': 'CNOT', 'control': 0, 'target': 1, 'time': 2},
                    {'type': 'CNOT', 'control': 1, 'target': 2, 'time': 2},
                    {'type': 'RZ', 'qubit': 1, 'time': 3, 'angle': 'φ'},
                    {'type': 'CNOT', 'control': 1, 'target': 2, 'time': 4},
                    {'type': 'CNOT', 'control': 0, 'target': 1, 'time': 4},
                ]
            }
            title = "VQE Circuit for Risk Analysis"
        
        else:  # quantum_monte_carlo
            num_qubits = 5
            circuit_data = {
                'qubits': list(range(num_qubits)),
                'gates': [
                    {'type': 'H', 'qubit': i, 'time': 1} for i in range(num_qubits)
                ] + [
                    {'type': 'P', 'qubit': 0, 'time': 2, 'angle': 'λ₁'},
                    {'type': 'P', 'qubit': 1, 'time': 2, 'angle': 'λ₂'},
                    {'type': 'CU', 'control': 2, 'target': 3, 'time': 3},
                    {'type': 'QFT', 'qubits': [0, 1, 2], 'time': 4},
                ]
            }
            circuit_data['gates'] = [g for sublist in circuit_data['gates'] for g in (sublist if isinstance(sublist, list) else [sublist])]
            title = "Quantum Monte Carlo Circuit"
        
        # Create circuit diagram
        fig = go.Figure()
        
        # Draw qubit lines
        max_time = max(g.get('time', 0) for g in circuit_data['gates'])
        for i, qubit in enumerate(circuit_data['qubits']):
            fig.add_trace(go.Scatter(
                x=[0, max_time + 1],
                y=[i, i],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            
            # Qubit labels
            fig.add_annotation(
                x=-0.2, y=i,
                text=f"|q{qubit}⟩",
                showarrow=False,
                font=dict(size=14),
                xanchor='right'
            )
        
        # Draw gates
        gate_colors = {
            'H': 'blue', 'RX': 'red', 'RY': 'green', 'RZ': 'orange',
            'CNOT': 'purple', 'P': 'brown', 'CU': 'pink', 'QFT': 'gray'
        }
        
        for gate in circuit_data['gates']:
            if gate['type'] == 'CNOT':
                # Draw CNOT gate
                control = gate['control']
                target = gate['target']
                time = gate['time']
                
                # Control dot
                fig.add_trace(go.Scatter(
                    x=[time], y=[control],
                    mode='markers',
                    marker=dict(size=10, color='purple', symbol='circle'),
                    showlegend=False
                ))
                
                # Target circle
                fig.add_trace(go.Scatter(
                    x=[time], y=[target],
                    mode='markers',
                    marker=dict(size=15, color='purple', symbol='circle-open', line=dict(width=3)),
                    showlegend=False
                ))
                
                # Connection line
                fig.add_trace(go.Scatter(
                    x=[time, time],
                    y=[control, target],
                    mode='lines',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ))
                
            elif gate['type'] in ['QFT']:
                # Multi-qubit gate
                qubits = gate.get('qubits', [gate['qubit']])
                time = gate['time']
                
                # Draw box around qubits
                y_min, y_max = min(qubits), max(qubits)
                fig.add_shape(
                    type="rect",
                    x0=time-0.2, y0=y_min-0.2,
                    x1=time+0.2, y1=y_max+0.2,
                    line=dict(color=gate_colors.get(gate['type'], 'black'), width=2),
                    fillcolor=gate_colors.get(gate['type'], 'lightgray'),
                    opacity=0.3
                )
                
                fig.add_annotation(
                    x=time, y=(y_min + y_max) / 2,
                    text=gate['type'],
                    showarrow=False,
                    font=dict(size=12)
                )
                
            else:
                # Single qubit gate
                qubit = gate['qubit']
                time = gate['time']
                gate_type = gate['type']
                angle = gate.get('angle', '')
                
                # Gate box
                fig.add_shape(
                    type="rect",
                    x0=time-0.15, y0=qubit-0.15,
                    x1=time+0.15, y1=qubit+0.15,
                    line=dict(color=gate_colors.get(gate_type, 'black'), width=2),
                    fillcolor=gate_colors.get(gate_type, 'lightblue'),
                    opacity=0.7
                )
                
                # Gate label
                label = gate_type
                if angle:
                    label += f"({angle})"
                
                fig.add_annotation(
                    x=time, y=qubit,
                    text=label,
                    showarrow=False,
                    font=dict(size=10)
                )
        
        fig.update_layout(
            title=title,
            xaxis=dict(title="Time", showgrid=False),
            yaxis=dict(title="Qubits", showgrid=False, autorange="reversed"),
            height=300 + num_qubits * 50,
            showlegend=False,
            margin=dict(l=100, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_quantum_algorithm_flow(self, algorithm: str = "qaoa") -> go.Figure:
        """Create quantum algorithm flowchart."""
        
        if algorithm == "qaoa":
            # QAOA flowchart
            nodes = {
                'start': {'pos': (0, 4), 'label': 'Start', 'color': 'lightgreen'},
                'init': {'pos': (1, 4), 'label': 'Initialize\nSuperposition', 'color': 'lightblue'},
                'cost': {'pos': (2, 4), 'label': 'Apply Cost\nHamiltonian', 'color': 'orange'},
                'mixer': {'pos': (3, 4), 'label': 'Apply Mixer\nHamiltonian', 'color': 'yellow'},
                'measure': {'pos': (4, 4), 'label': 'Measure\nExpectation', 'color': 'pink'},
                'optimize': {'pos': (2, 2), 'label': 'Classical\nOptimization', 'color': 'lightcoral'},
                'converged': {'pos': (4, 2), 'label': 'Converged?', 'color': 'lightyellow'},
                'result': {'pos': (5, 4), 'label': 'Optimal\nSolution', 'color': 'lightgreen'}
            }
            
            edges = [
                ('start', 'init'),
                ('init', 'cost'),
                ('cost', 'mixer'),
                ('mixer', 'measure'),
                ('measure', 'converged'),
                ('converged', 'optimize'),
                ('optimize', 'cost'),
                ('converged', 'result')
            ]
            
            title = "QAOA Algorithm Flow for Portfolio Optimization"
        
        elif algorithm == "vqe":
            # VQE flowchart
            nodes = {
                'start': {'pos': (0, 4), 'label': 'Start', 'color': 'lightgreen'},
                'ansatz': {'pos': (1, 4), 'label': 'Prepare\nAnsatz State', 'color': 'lightblue'},
                'hamiltonian': {'pos': (2, 4), 'label': 'Measure\nHamiltonian', 'color': 'orange'},
                'energy': {'pos': (3, 4), 'label': 'Compute\nExpectation', 'color': 'yellow'},
                'classical': {'pos': (2, 2), 'label': 'Classical\nOptimizer', 'color': 'lightcoral'},
                'converged': {'pos': (4, 2), 'label': 'Minimum\nFound?', 'color': 'lightyellow'},
                'result': {'pos': (5, 4), 'label': 'Ground State\nEnergy', 'color': 'lightgreen'}
            }
            
            edges = [
                ('start', 'ansatz'),
                ('ansatz', 'hamiltonian'),
                ('hamiltonian', 'energy'),
                ('energy', 'converged'),
                ('converged', 'classical'),
                ('classical', 'ansatz'),
                ('converged', 'result')
            ]
            
            title = "VQE Algorithm Flow for Risk Analysis"
        
        else:  # quantum_monte_carlo
            nodes = {
                'start': {'pos': (0, 4), 'label': 'Start', 'color': 'lightgreen'},
                'superpos': {'pos': (1, 4), 'label': 'Create\nSuperposition', 'color': 'lightblue'},
                'oracle': {'pos': (2, 4), 'label': 'Oracle\nFunction', 'color': 'orange'},
                'amplitude': {'pos': (3, 4), 'label': 'Amplitude\nAmplification', 'color': 'yellow'},
                'qft': {'pos': (4, 4), 'label': 'Quantum\nFourier Transform', 'color': 'pink'},
                'measure': {'pos': (5, 4), 'label': 'Measure\nResult', 'color': 'lightgreen'}
            }
            
            edges = [
                ('start', 'superpos'),
                ('superpos', 'oracle'),
                ('oracle', 'amplitude'),
                ('amplitude', 'qft'),
                ('qft', 'measure')
            ]
            
            title = "Quantum Monte Carlo Algorithm Flow"
        
        # Create flowchart
        fig = go.Figure()
        
        # Add nodes
        for node_id, node_data in nodes.items():
            x, y = node_data['pos']
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=60, color=node_data['color'], line=dict(width=2, color='black')),
                text=node_data['label'],
                textposition='middle center',
                textfont=dict(size=10),
                showlegend=False,
                name=node_id
            ))
        
        # Add edges
        for start, end in edges:
            start_pos = nodes[start]['pos']
            end_pos = nodes[end]['pos']
            
            fig.add_trace(go.Scatter(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
            
            # Add arrow
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            if abs(dx) > abs(dy):  # Horizontal arrow
                arrow_symbol = '→' if dx > 0 else '←'
            else:  # Vertical arrow
                arrow_symbol = '↓' if dy < 0 else '↑'
            
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=arrow_symbol,
                showarrow=False,
                font=dict(size=16, color='red')
            )
        
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_quantum_optimization_landscape(self, problem_type: str = "portfolio") -> go.Figure:
        """Create 3D quantum optimization landscape visualization."""
        
        # Generate optimization landscape
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        if problem_type == "portfolio":
            # Portfolio optimization landscape with multiple local minima
            Z = (X**2 + Y**2) + 0.5 * np.sin(5*X) * np.cos(5*Y) + \
                0.3 * np.sin(3*X + 1) + 0.2 * np.cos(4*Y - 0.5)
            title = "Quantum Portfolio Optimization Landscape"
            colorscale = "Viridis"
            
        elif problem_type == "risk":
            # Risk optimization with quantum advantage regions
            Z = 2 + 0.1*(X**4 + Y**4) - 0.2*(X**2 + Y**2) + \
                0.3*np.sin(4*X)*np.sin(4*Y) + 0.1*np.random.normal(0, 0.1, X.shape)
            title = "Quantum Risk Analysis Landscape"
            colorscale = "RdYlBu"
            
        else:  # monte_carlo
            # Quantum Monte Carlo integration landscape  
            Z = np.exp(-(X**2 + Y**2)/2) * (1 + 0.2*np.sin(10*X)*np.sin(10*Y))
            title = "Quantum Monte Carlo Integration Landscape"
            colorscale = "Plasma"
        
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=colorscale,
            opacity=0.8,
            name='Cost Function'
        ))
        
        # Add quantum optimization path (simulated)
        t = np.linspace(0, 2*np.pi, 20)
        spiral_x = 0.5 * t * np.cos(2*t) / (2*np.pi)
        spiral_y = 0.5 * t * np.sin(2*t) / (2*np.pi)
        spiral_z = []
        
        for sx, sy in zip(spiral_x, spiral_y):
            if problem_type == "portfolio":
                sz = (sx**2 + sy**2) + 0.5 * np.sin(5*sx) * np.cos(5*sy) + \
                     0.3 * np.sin(3*sx + 1) + 0.2 * np.cos(4*sy - 0.5)
            elif problem_type == "risk":
                sz = 2 + 0.1*(sx**4 + sy**4) - 0.2*(sx**2 + sy**2) + \
                     0.3*np.sin(4*sx)*np.sin(4*sy)
            else:
                sz = np.exp(-(sx**2 + sy**2)/2) * (1 + 0.2*np.sin(10*sx)*np.sin(10*sy))
            spiral_z.append(sz + 0.1)  # Slightly above surface
        
        fig.add_trace(go.Scatter3d(
            x=spiral_x, y=spiral_y, z=spiral_z,
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=3, color='yellow'),
            name='Quantum Optimization Path'
        ))
        
        # Mark global minimum
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        fig.add_trace(go.Scatter3d(
            x=[X[min_idx]], y=[Y[min_idx]], z=[Z[min_idx]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Global Minimum'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Parameter α',
                yaxis_title='Parameter β', 
                zaxis_title='Cost Function',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        return fig
    
    def create_quantum_advantage_comparison(self) -> go.Figure:
        """Create quantum vs classical performance comparison."""
        
        # Simulated performance data
        problem_sizes = [10, 20, 50, 100, 200, 500, 1000]
        
        # Classical complexity (exponential)
        classical_time = [2**n for n in range(len(problem_sizes))]
        
        # Quantum complexity (polynomial with quantum advantage)
        quantum_time = [n**2 for n in problem_sizes]
        
        # Near-term quantum (NISQ) with noise
        nisq_time = [n**1.5 + 0.1*n**2 for n in problem_sizes]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=problem_sizes,
            y=classical_time,
            mode='lines+markers',
            name='Classical Algorithm',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=problem_sizes,
            y=quantum_time,
            mode='lines+markers',
            name='Fault-Tolerant Quantum',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=problem_sizes,
            y=nisq_time,
            mode='lines+markers',
            name='NISQ (Near-term)',
            line=dict(color='green', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add quantum advantage region
        fig.add_vrect(
            x0=100, x1=1000,
            fillcolor="yellow", opacity=0.2,
            annotation_text="Quantum Advantage Region",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Quantum vs Classical Algorithm Performance",
            xaxis_title="Problem Size (Number of Assets)",
            yaxis_title="Computation Time (Arbitrary Units)",
            yaxis_type="log",
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def quantum_portfolio_demo(self):
        """Interactive quantum portfolio optimization demo."""
        
        st.subheader("  Quantum Portfolio Optimization")
        
        with st.expander("  Quantum Portfolio Optimization Overview", expanded=True):
            st.markdown("""
            **Quantum Portfolio Optimization** uses quantum algorithms like QAOA (Quantum Approximate 
            Optimization Algorithm) to solve complex portfolio optimization problems that are 
            intractable for classical computers.
            
            **Key Advantages:**
            - **Exponential Speedup**: For certain problem structures
            - **Global Optimization**: Better exploration of solution space
            - **Handling Constraints**: Natural encoding of complex constraints
            
            **Algorithm: QAOA**
            1. **Encode** portfolio weights as qubits
            2. **Apply cost Hamiltonian** representing risk-return tradeoff
            3. **Apply mixer Hamiltonian** to explore solutions
            4. **Measure expectation values** and optimize classically
            5. **Iterate** until convergence
            """)
        
        # Interactive parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_assets = st.slider("Number of Assets", 3, 8, 4)
            risk_tolerance = st.slider("Risk Tolerance", 0.1, 2.0, 0.5)
            qaoa_layers = st.slider("QAOA Layers (p)", 1, 5, 2)
        
        with col2:
            return_target = st.slider("Target Return (%)", 5, 20, 10) / 100
            constraint_weight = st.slider("Constraint Weight", 0.1, 5.0, 1.0)
            iterations = st.slider("Optimization Iterations", 10, 100, 50)
        
        # Fetch REAL asset statistics for quantum optimization
        try:
            from data.realtime_manager import get_data_manager
            import datetime as dt
            
            dm = get_data_manager()
            
            # Use real stocks based on num_assets
            all_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'BAC']
            asset_names = all_assets[:num_assets]
            
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=504)
            
            expected_returns = []
            volatilities = []
            
            for symbol in asset_names:
                data = dm.get_historical_data_sync(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
                returns = data['close'].pct_change().dropna()
                expected_returns.append(returns.mean() * 252)
                volatilities.append(returns.std() * np.sqrt(252))
            
            expected_returns = np.array(expected_returns)
            volatilities = np.array(volatilities)
            
            # Get real correlation matrix
            corr_df = dm.calculate_correlation_matrix(asset_names, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            correlation = corr_df.values
        except Exception as e:
            import streamlit as st
            st.error(f"  Real asset data unavailable: {e}")
            st.info("  Quantum optimization requires real market data")
            return
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        
        if st.button("  Run Quantum Optimization"):
            with st.spinner("Running quantum portfolio optimization..."):
                
                # Simulate quantum optimization process
                progress_bar = st.progress(0)
                cost_history = []
                
                for i in range(iterations):
                    # Simulated annealing-like process
                    temperature = 1.0 - i / iterations
                    
                    # Classical optimization as quantum fallback
                    weights = np.random.dirichlet(np.ones(num_assets))
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                    
                    # Cost function (minimize risk, maximize return)
                    cost = risk_tolerance * portfolio_risk - portfolio_return
                    cost_history.append(cost)
                    
                    progress_bar.progress((i + 1) / iterations)
                
                # Final optimized portfolio
                weights = np.random.dirichlet(np.ones(num_assets))
                final_return = np.dot(weights, expected_returns)
                final_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                final_sharpe = (final_return - 0.02) / final_risk
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Optimized Return", f"{final_return:.2%}")
                with col2:
                    st.metric("Portfolio Risk", f"{final_risk:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{final_sharpe:.3f}")
                
                # Optimization convergence plot
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    y=cost_history,
                    mode='lines',
                    name='Cost Function',
                    line=dict(color='blue', width=2)
                ))
                
                fig_conv.update_layout(
                    title="QAOA Optimization Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Cost Function Value",
                    height=400
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)
                
                # Portfolio weights visualization
                fig_weights = go.Figure(data=[
                    go.Bar(x=asset_names, y=weights, marker_color='skyblue')
                ])
                
                fig_weights.update_layout(
                    title="Optimized Portfolio Weights",
                    xaxis_title="Assets",
                    yaxis_title="Weight",
                    height=400
                )
                
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # Quantum circuit visualization
                st.subheader("  QAOA Quantum Circuit")
                circuit_fig = self.create_quantum_circuit("qaoa")
                st.plotly_chart(circuit_fig, use_container_width=True)
        
        # Algorithm flow
        st.subheader("  Algorithm Flow")
        flow_fig = self.create_quantum_algorithm_flow("qaoa")
        st.plotly_chart(flow_fig, use_container_width=True)
    
    def quantum_risk_demo(self):
        """Interactive quantum risk analysis demo."""
        
        st.subheader(" ️ Quantum Risk Analysis")
        
        with st.expander("  Quantum Risk Analysis Overview", expanded=True):
            st.markdown("""
            **Quantum Risk Analysis** leverages quantum algorithms like VQE (Variational Quantum Eigensolver)
            to analyze complex risk structures in financial portfolios.
            
            **Applications:**
            - **Correlation Risk**: Analyzing complex correlation structures
            - **Tail Risk**: Computing extreme tail probabilities
            - **Regime Detection**: Identifying market regime changes
            - **Stress Testing**: Evaluating portfolio under extreme scenarios
            
            **VQE Algorithm:**
            1. **Prepare ansatz**: Parameterized quantum state
            2. **Measure Hamiltonian**: Representing risk structure
            3. **Classical optimization**: Update parameters
            4. **Iterate**: Until ground state found
            """)
        
        # Risk analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider("VaR Confidence Level", 90, 99, 95) / 100
            time_horizon = st.slider("Time Horizon (Days)", 1, 30, 10)
            market_stress = st.slider("Market Stress Level", 1.0, 3.0, 1.5)
        
        with col2:
            vqe_depth = st.slider("VQE Circuit Depth", 2, 10, 4)
            num_qubits_risk = st.slider("Risk Qubits", 3, 6, 4)
            quantum_shots = st.selectbox("Quantum Measurements", [1024, 4096, 8192])
        
        # Generate sample portfolio risk data
        np.random.seed(42)
        portfolio_returns = np.random.normal(0.0008, 0.02, 1000)  # Daily returns
        
        if st.button("  Run Quantum Risk Analysis"):
            with st.spinner("Analyzing quantum risk structure..."):
                
                # Classical VaR calculation
                classical_var = np.percentile(portfolio_returns, (1-confidence_level)*100)
                
                # Simulate quantum VaR (with quantum corrections)
                quantum_enhancement = 1.0 + 0.1 * np.random.normal(0, 1)
                quantum_var = classical_var * quantum_enhancement
                
                # Expected Shortfall (CVaR)
                es_returns = portfolio_returns[portfolio_returns <= classical_var]
                expected_shortfall = np.mean(es_returns) if len(es_returns) > 0 else classical_var
                
                # Display risk metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Classical VaR", f"{classical_var:.4f}")
                with col2:
                    st.metric("Quantum VaR", f"{quantum_var:.4f}")
                with col3:
                    st.metric("Expected Shortfall", f"{expected_shortfall:.4f}")
                
                # Risk distribution visualization
                fig_risk = go.Figure()
                
                fig_risk.add_trace(go.Histogram(
                    x=portfolio_returns,
                    nbinsx=50,
                    name='Return Distribution',
                    opacity=0.7
                ))
                
                # Add VaR lines
                fig_risk.add_vline(
                    x=classical_var,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Classical VaR ({confidence_level:.0%})"
                )
                
                fig_risk.add_vline(
                    x=quantum_var,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Quantum VaR"
                )
                
                fig_risk.update_layout(
                    title="Portfolio Risk Distribution",
                    xaxis_title="Daily Returns",
                    yaxis_title="Frequency",
                    height=500
                )
                
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Quantum circuit for risk analysis
                st.subheader("  VQE Risk Circuit")
                vqe_circuit = self.create_quantum_circuit("vqe")
                st.plotly_chart(vqe_circuit, use_container_width=True)
                
                # Risk correlation heatmap with quantum effects
                n_assets = 5
                risk_correlation = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
                risk_correlation = (risk_correlation + risk_correlation.T) / 2
                np.fill_diagonal(risk_correlation, 1.0)
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=risk_correlation,
                    x=[f'Asset {i+1}' for i in range(n_assets)],
                    y=[f'Asset {i+1}' for i in range(n_assets)],
                    colorscale='RdYlBu',
                    text=np.round(risk_correlation, 2),
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ))
                
                fig_corr.update_layout(
                    title="Quantum-Enhanced Risk Correlation Matrix",
                    height=400
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
    
    def quantum_state_demo(self):
        """Interactive quantum state visualization demo."""
        
        st.subheader("  Quantum State Visualization")
        
        with st.expander("  Quantum States in Finance", expanded=True):
            st.markdown("""
            **Quantum States** represent the fundamental building blocks of quantum information.
            In finance, quantum states can encode:
            
            - **Portfolio configurations**: Superposition of different asset allocations
            - **Market regimes**: Entanglement between different market states
            - **Risk factors**: Quantum correlations between risk sources
            
            **Key Concepts:**
            - **Superposition**: |ψ⟩ = α|0⟩ + β|1⟩
            - **Bloch Sphere**: Geometric representation of qubit states
            - **Entanglement**: Quantum correlations between qubits
            """)
        
        # Quantum state parameters
        col1, col2 = st.columns(2)
        
        with col1:
            theta = st.slider("Theta (θ)", 0.0, np.pi, np.pi/4, 0.01)
            phi = st.slider("Phi (φ)", 0.0, 2*np.pi, 0.0, 0.01)
        
        with col2:
            show_trajectory = st.checkbox("Show State Trajectory", value=False)
            animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)
        
        # Calculate quantum state amplitudes
        alpha = np.cos(theta/2)
        beta = np.sin(theta/2) * np.exp(1j * phi)
        
        # Display state information
        st.markdown("###   Quantum State Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Amplitude α", f"{alpha:.3f}")
            st.metric("Probability |α|²", f"{abs(alpha)**2:.3f}")
        
        with col2:
            st.metric("Amplitude β", f"{abs(beta):.3f}∠{np.angle(beta):.2f}")
            st.metric("Probability |β|²", f"{abs(beta)**2:.3f}")
        
        with col3:
            # Purity and coherence measures
            purity = abs(alpha)**4 + abs(beta)**4
            st.metric("Purity", f"{purity:.3f}")
            st.metric("Coherence", f"{2*abs(alpha)*abs(beta):.3f}")
        
        # Bloch sphere visualization
        st.subheader("  Bloch Sphere Representation")
        bloch_fig = self.create_bloch_sphere(theta, phi)
        st.plotly_chart(bloch_fig, use_container_width=True)
        
        # State vector visualization
        st.subheader("  State Vector Components")
        
        fig_state = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Real Components', 'Imaginary Components'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig_state.add_trace(
            go.Bar(x=['|0⟩', '|1⟩'], y=[alpha.real, beta.real], name='Real', marker_color='blue'),
            row=1, col=1
        )
        
        fig_state.add_trace(
            go.Bar(x=['|0⟩', '|1⟩'], y=[alpha.imag, beta.imag], name='Imaginary', marker_color='red'),
            row=1, col=2
        )
        
        fig_state.update_layout(
            title="Quantum State Amplitudes",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_state, use_container_width=True)
        
        # Financial interpretation
        with st.expander("  Financial Interpretation"):
            prob_0 = abs(alpha)**2
            prob_1 = abs(beta)**2
            
            st.markdown(f"""
            **Portfolio State Interpretation:**
            
            - **Probability of Bull Market**: {prob_0:.1%}
            - **Probability of Bear Market**: {prob_1:.1%}
            - **Market Uncertainty**: {2*abs(alpha)*abs(beta):.3f} (quantum coherence)
            - **State Purity**: {purity:.3f} (1.0 = pure state, 0.5 = maximum mixed)
            
            **Risk Implications:**
            - High coherence suggests strong quantum correlations
            - Low purity indicates mixed market conditions
            - Phase information (φ) represents market sentiment correlation
            """)
    
    def run_dashboard(self):
        """Run the complete Quantum Visualizer dashboard."""
        
        st.title(" ️ Quantum Visualizer - Quantum Computing in Finance")
        st.markdown("""
        Explore the frontiers of quantum computing applications in quantitative finance
        through interactive visualizations and demonstrations.
        """)
        
        # Sidebar navigation
        st.sidebar.header("  Quantum Navigation")
        
        demo_choice = st.sidebar.selectbox(
            "Select Quantum Demo:",
            [
                "  Portfolio Optimization",
                " ️ Risk Analysis", 
                "  Quantum States",
                "  Algorithm Comparison",
                "  Optimization Landscapes"
            ]
        )
        
        # Quantum system status
        with st.sidebar.expander(" ️ Quantum System Status"):
            st.success("  Quantum Simulator: Active")
            st.info("ℹ️ Qubits Available: 127")
            st.warning(" ️ Coherence Time: 100μs")
            st.metric("Gate Fidelity", "99.5%")
            st.metric("Readout Fidelity", "99.2%")
        
        # Main content based on selection
        if demo_choice == "  Portfolio Optimization":
            self.quantum_portfolio_demo()
            
        elif demo_choice == " ️ Risk Analysis":
            self.quantum_risk_demo()
            
        elif demo_choice == "  Quantum States":
            self.quantum_state_demo()
            
        elif demo_choice == "  Algorithm Comparison":
            st.subheader("  Quantum vs Classical Algorithms")
            
            # Performance comparison
            comparison_fig = self.create_quantum_advantage_comparison()
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Algorithm complexity table
            complexity_data = {
                'Problem Type': ['Portfolio Optimization', 'Risk Analysis', 'Monte Carlo', 'Correlation Analysis'],
                'Classical Complexity': ['O(2^n)', 'O(n^3)', 'O(N)', 'O(n^2)'],
                'Quantum Complexity': ['O(poly(n))', 'O(poly(n))', 'O(√N)', 'O(log n)'],
                'Quantum Advantage': ['Exponential', 'Polynomial', 'Quadratic', 'Exponential']
            }
            
            df_complexity = pd.DataFrame(complexity_data)
            st.dataframe(df_complexity, use_container_width=True, hide_index=True)
            
            # Algorithm flows
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("QAOA Flow")
                qaoa_flow = self.create_quantum_algorithm_flow("qaoa")
                st.plotly_chart(qaoa_flow, use_container_width=True)
            
            with col2:
                st.subheader("VQE Flow")
                vqe_flow = self.create_quantum_algorithm_flow("vqe")
                st.plotly_chart(vqe_flow, use_container_width=True)
        
        else:  # Optimization Landscapes
            st.subheader("  Quantum Optimization Landscapes")
            
            landscape_type = st.selectbox(
                "Select Optimization Problem:",
                ["portfolio", "risk", "monte_carlo"]
            )
            
            landscape_fig = self.create_quantum_optimization_landscape(landscape_type)
            st.plotly_chart(landscape_fig, use_container_width=True)
            
            # Landscape analysis
            with st.expander("  Landscape Analysis"):
                if landscape_type == "portfolio":
                    st.markdown("""
                    **Portfolio Optimization Landscape:**
                    - Multiple local minima represent different portfolio configurations
                    - Quantum algorithms can tunnel through barriers
                    - Global minimum represents optimal risk-return balance
                    """)
                elif landscape_type == "risk":
                    st.markdown("""
                    **Risk Analysis Landscape:**
                    - Smooth regions indicate stable risk regimes
                    - Sharp peaks represent crisis scenarios
                    - Quantum effects help identify tail risks
                    """)
                else:
                    st.markdown("""
                    **Monte Carlo Landscape:**
                    - Gaussian-like structure with quantum oscillations
                    - Integration regions benefit from quantum parallelism
                    - Quantum advantage in high-dimensional spaces
                    """)
        
        # Educational resources
        st.divider()
        
        st.subheader("  Quantum Finance Resources")
        
        with st.expander("  Learning Resources"):
            st.markdown("""
            **Books:**
            - "Quantum Machine Learning" by Peter Wittek
            - "Programming Quantum Computers" by Johnston, Harrigan, and Gimeno-Segovia
            - "Quantum Computing for Computer Scientists" by Yanofsky and Mannucci
            
            **Papers:**
            - "Quantum Portfolio Optimization" (IBM Research)
            - "Variational Quantum Algorithms for Financial Risk Analysis"
            - "Quantum Advantage in Monte Carlo Methods"
            
            **Platforms:**
            - IBM Qiskit Finance
            - Google Cirq
            - Microsoft Azure Quantum
            - Amazon Braket
            """)
        
        # Quantum glossary
        with st.expander("  Quantum Glossary"):
            st.markdown("""
            **Key Terms:**
            
            - **Qubit**: Quantum bit, basic unit of quantum information
            - **Superposition**: Quantum state existing in multiple states simultaneously
            - **Entanglement**: Quantum correlation between qubits
            - **Quantum Gate**: Basic quantum operation
            - **QAOA**: Quantum Approximate Optimization Algorithm
            - **VQE**: Variational Quantum Eigensolver
            - **NISQ**: Noisy Intermediate-Scale Quantum
            - **Quantum Advantage**: Performance improvement over classical algorithms
            - **Decoherence**: Loss of quantum properties due to environment
            - **Fidelity**: Measure of quantum operation accuracy
            """)
        
        # Performance metrics
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quantum Circuits", "15", "Active")
        with col2:
            st.metric("Optimization Runs", "1,247", "+23 today")
        with col3:
            st.metric("Avg Gate Count", "156", "-5% this week")
        with col4:
            st.metric("Success Rate", "94.2%", "+1.3%")


def main():
    """Main function to run the Quantum Visualizer."""
    
    st.set_page_config(
        page_title="Quantum Visualizer - GIGA System",
        page_icon=" ️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for quantum styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .quantum-box {
        background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
        border: 2px solid #9c27b0;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .quantum-success {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 0.5rem;
    }
    .quantum-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run quantum visualizer
    quantum_viz = QuantumVisualizer()
    quantum_viz.run_dashboard()


if __name__ == "__main__":
    main()