"""
Greeks Dashboard - 3D Interactive Visualization
==============================================

Advanced 3D visualization dashboard for option Greeks using Plotly.
Provides interactive surfaces, contour plots, and real-time Greek analysis.

Features:
- 3D Greek surfaces (Delta, Gamma, Theta, Vega)
- Interactive parameter controls
- Cross-sectional analysis
- Real-time calculations
- Professional styling
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from core.black_scholes import BlackScholesCalculator
    from core.greeks import GreeksCalculator
    from utils.performance_profiler import PerformanceProfiler
    from data.realtime_manager import get_data_manager, get_realtime_price
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
    # Fallback implementations
    class BlackScholesCalculator:
        def call_price(self, S, K, T, r, sigma):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        def put_price(self, S, K, T, r, sigma):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    class GreeksCalculator:
        def __init__(self):
            self.bs = BlackScholesCalculator()
        
        def delta(self, S, K, T, r, sigma, option_type='call'):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            if option_type == 'call':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1
        
        def gamma(self, S, K, T, r, sigma, option_type='call'):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        def theta(self, S, K, T, r, sigma, option_type='call'):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            theta_time = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            
            if option_type == 'call':
                theta_rate = -r * K * np.exp(-r*T) * norm.cdf(d2)
                return (theta_time + theta_rate) / 365
            else:
                theta_rate = r * K * np.exp(-r*T) * norm.cdf(-d2)
                return (theta_time + theta_rate) / 365
        
        def vega(self, S, K, T, r, sigma, option_type='call'):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start


class GreeksDashboard:
    """Interactive 3D Greeks visualization dashboard."""
    
    def __init__(self):
        """Initialize Greeks dashboard."""
        self.bs_calc = BlackScholesCalculator()
        self.greeks_calc = GreeksCalculator()
        self.profiler = PerformanceProfiler()
        
        # Default parameters
        self.default_params = {
            'spot_price': 100.0,
            'strike_price': 100.0,
            'time_to_expiry': 0.25,
            'risk_free_rate': 0.05,
            'volatility': 0.20
        }
        
        # Greek color schemes
        self.greek_colors = {
            'delta': 'viridis',
            'gamma': 'plasma',
            'theta': 'inferno',
            'vega': 'cividis',
            'rho': 'turbo'
        }
    
    def create_parameter_controls(self) -> Dict:
        """Create interactive parameter controls."""
        st.sidebar.header("  Option Parameters")
        
        params = {}
        
        # Current spot price
        params['spot_price'] = st.sidebar.slider(
            "Spot Price ($)",
            min_value=50.0,
            max_value=200.0,
            value=self.default_params['spot_price'],
            step=1.0,
            help="Current price of the underlying asset"
        )
        
        # Strike price
        params['strike_price'] = st.sidebar.slider(
            "Strike Price ($)",
            min_value=50.0,
            max_value=200.0,
            value=self.default_params['strike_price'],
            step=1.0,
            help="Strike price of the option"
        )
        
        # Time to expiry
        params['time_to_expiry'] = st.sidebar.slider(
            "Time to Expiry (Years)",
            min_value=0.01,
            max_value=2.0,
            value=self.default_params['time_to_expiry'],
            step=0.01,
            help="Time remaining until option expiration"
        )
        
        # Risk-free rate
        params['risk_free_rate'] = st.sidebar.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=self.default_params['risk_free_rate'] * 100,
            step=0.1,
            help="Risk-free interest rate"
        ) / 100
        
        # Volatility
        params['volatility'] = st.sidebar.slider(
            "Volatility (%)",
            min_value=5.0,
            max_value=100.0,
            value=self.default_params['volatility'] * 100,
            step=1.0,
            help="Implied volatility of the underlying"
        ) / 100
        
        # Option type
        params['option_type'] = st.sidebar.selectbox(
            "Option Type",
            ['call', 'put'],
            help="Type of option contract"
        )
        
        return params
    
    def generate_3d_surface_data(self, greek_func, params: Dict, 
                                spot_range: Tuple[float, float] = (70, 130),
                                vol_range: Tuple[float, float] = (0.1, 0.5),
                                resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D surface data for Greek visualization."""
        
        # Create meshgrid for spot prices and volatilities
        spot_prices = np.linspace(spot_range[0], spot_range[1], resolution)
        volatilities = np.linspace(vol_range[0], vol_range[1], resolution)
        S_mesh, Vol_mesh = np.meshgrid(spot_prices, volatilities)
        
        # Calculate Greek values
        greek_values = np.zeros_like(S_mesh)
        
        for i in range(len(volatilities)):
            for j in range(len(spot_prices)):
                try:
                    greek_values[i, j] = greek_func(
                        S_mesh[i, j],
                        params['strike_price'],
                        params['time_to_expiry'],
                        params['risk_free_rate'],
                        Vol_mesh[i, j],
                        params['option_type']
                    )
                except:
                    greek_values[i, j] = np.nan
        
        return S_mesh, Vol_mesh, greek_values
    
    def create_3d_surface_plot(self, greek_name: str, params: Dict) -> go.Figure:
        """Create 3D surface plot for a specific Greek."""
        
        # Get the appropriate Greek function
        greek_funcs = {
            'delta': self.greeks_calc.delta,
            'gamma': self.greeks_calc.gamma,
            'theta': self.greeks_calc.theta,
            'vega': self.greeks_calc.vega
        }
        
        if greek_name not in greek_funcs:
            raise ValueError(f"Unknown Greek: {greek_name}")
        
        greek_func = greek_funcs[greek_name]
        
        # Generate surface data
        S_mesh, Vol_mesh, greek_values = self.generate_3d_surface_data(
            greek_func, params
        )
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=S_mesh,
                y=Vol_mesh * 100,  # Convert to percentage
                z=greek_values,
                colorscale=self.greek_colors[greek_name],
                name=greek_name.upper(),
                showscale=True,
                colorbar=dict(
                    title=f"{greek_name.upper()}",
                    titleside="right",
                    tickmode="linear",
                    tick0=np.nanmin(greek_values),
                    dtick=(np.nanmax(greek_values) - np.nanmin(greek_values)) / 10
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"3D {greek_name.upper()} Surface",
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title="Spot Price ($)",
                yaxis_title="Volatility (%)",
                zaxis_title=f"{greek_name.upper()}",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def create_greek_comparison_chart(self, params: Dict) -> go.Figure:
        """Create comparison chart of all Greeks."""
        
        # Calculate Greeks for different spot prices
        spot_prices = np.linspace(80, 120, 100)
        greeks_data = {
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': []
        }
        
        for S in spot_prices:
            greeks_data['delta'].append(
                self.greeks_calc.delta(S, params['strike_price'], params['time_to_expiry'],
                                     params['risk_free_rate'], params['volatility'], params['option_type'])
            )
            greeks_data['gamma'].append(
                self.greeks_calc.gamma(S, params['strike_price'], params['time_to_expiry'],
                                     params['risk_free_rate'], params['volatility'], params['option_type'])
            )
            greeks_data['theta'].append(
                self.greeks_calc.theta(S, params['strike_price'], params['time_to_expiry'],
                                     params['risk_free_rate'], params['volatility'], params['option_type'])
            )
            greeks_data['vega'].append(
                self.greeks_calc.vega(S, params['strike_price'], params['time_to_expiry'],
                                    params['risk_free_rate'], params['volatility'], params['option_type'])
            )
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Delta', 'Gamma', 'Theta', 'Vega'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=spot_prices, y=greeks_data['delta'], name='Delta', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=spot_prices, y=greeks_data['gamma'], name='Gamma',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=spot_prices, y=greeks_data['theta'], name='Theta',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=spot_prices, y=greeks_data['vega'], name='Vega',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Greeks Analysis vs Spot Price",
            height=600,
            showlegend=False
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Spot Price ($)")
        
        # Add current spot price line
        current_spot = params['spot_price']
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(
                    x=current_spot, 
                    line_dash="dash", 
                    line_color="gray",
                    row=row, col=col
                )
        
        return fig
    
    def create_heatmap_visualization(self, params: Dict) -> go.Figure:
        """Create heatmap visualization of Greeks."""
        
        # Generate data for heatmap
        spot_range = np.linspace(80, 120, 20)
        time_range = np.linspace(0.01, 0.5, 20)
        
        # Calculate Delta heatmap
        delta_matrix = np.zeros((len(time_range), len(spot_range)))
        
        for i, T in enumerate(time_range):
            for j, S in enumerate(spot_range):
                delta_matrix[i, j] = self.greeks_calc.delta(
                    S, params['strike_price'], T,
                    params['risk_free_rate'], params['volatility'], params['option_type']
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=spot_range,
            y=time_range,
            z=delta_matrix,
            colorscale='viridis',
            colorbar=dict(title="Delta")
        ))
        
        fig.update_layout(
            title="Delta Heatmap: Spot Price vs Time to Expiry",
            xaxis_title="Spot Price ($)",
            yaxis_title="Time to Expiry (Years)",
            height=500
        )
        
        return fig
    
    def display_greek_metrics(self, params: Dict):
        """Display current Greek values as metrics."""
        
        # Calculate current Greeks
        current_greeks = {}
        
        performance_times = {}
        
        for greek_name in ['delta', 'gamma', 'theta', 'vega']:
            greek_func = getattr(self.greeks_calc, greek_name)
            result, exec_time = self.profiler.profile_function(
                greek_func,
                params['spot_price'], params['strike_price'], params['time_to_expiry'],
                params['risk_free_rate'], params['volatility'], params['option_type']
            )
            current_greeks[greek_name] = result
            performance_times[greek_name] = exec_time * 1000  # Convert to ms
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Delta (Δ)",
                f"{current_greeks['delta']:.4f}",
                help="Price sensitivity to underlying movement"
            )
            st.caption(f" ️ {performance_times['delta']:.3f}ms")
        
        with col2:
            st.metric(
                "Gamma (Γ)",
                f"{current_greeks['gamma']:.4f}",
                help="Rate of change of Delta"
            )
            st.caption(f" ️ {performance_times['gamma']:.3f}ms")
        
        with col3:
            st.metric(
                "Theta (Θ)",
                f"{current_greeks['theta']:.4f}",
                help="Time decay per day"
            )
            st.caption(f" ️ {performance_times['theta']:.3f}ms")
        
        with col4:
            st.metric(
                "Vega (ν)",
                f"{current_greeks['vega']:.4f}",
                help="Sensitivity to volatility changes"
            )
            st.caption(f" ️ {performance_times['vega']:.3f}ms")
    
    def display_option_value(self, params: Dict):
        """Display current option value."""
        
        if params['option_type'] == 'call':
            price, exec_time = self.profiler.profile_function(
                self.bs_calc.call_price,
                params['spot_price'], params['strike_price'], params['time_to_expiry'],
                params['risk_free_rate'], params['volatility']
            )
        else:
            price, exec_time = self.profiler.profile_function(
                self.bs_calc.put_price,
                params['spot_price'], params['strike_price'], params['time_to_expiry'],
                params['risk_free_rate'], params['volatility']
            )
        
        st.subheader(f"  {params['option_type'].title()} Option Value")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Option Price", f"${price:.4f}")
        
        with col2:
            intrinsic = max(0, params['spot_price'] - params['strike_price']) if params['option_type'] == 'call' else max(0, params['strike_price'] - params['spot_price'])
            st.metric("Intrinsic Value", f"${intrinsic:.4f}")
        
        with col3:
            time_value = price - intrinsic
            st.metric("Time Value", f"${time_value:.4f}")
        
        st.caption(f" ️ Calculation time: {exec_time*1000:.3f}ms")
    
    def run_dashboard(self):
        """Run the complete Greeks dashboard."""
        
        st.title("  Greeks Dashboard - 3D Interactive Visualization")
        st.markdown("""
        Explore option Greeks through interactive 3D surfaces, real-time calculations, 
        and comprehensive analysis tools.
        """)
        
        # Create parameter controls
        params = self.create_parameter_controls()
        
        # Display current option value
        self.display_option_value(params)
        
        st.divider()
        
        # Display current Greek metrics
        st.subheader("  Current Greek Values")
        self.display_greek_metrics(params)
        
        st.divider()
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["3D Surfaces", "Greek Comparison", "Heatmap", "Analysis"])
        
        with tab1:
            st.subheader("  3D Greek Surfaces")
            
            selected_greek = st.selectbox(
                "Select Greek for 3D Visualization",
                ['delta', 'gamma', 'theta', 'vega']
            )
            
            with st.spinner(f"Generating 3D {selected_greek.upper()} surface..."):
                fig_3d = self.create_3d_surface_plot(selected_greek, params)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.info("  **Tip**: Drag to rotate, scroll to zoom, click and drag to pan")
        
        with tab2:
            st.subheader("  Greek Comparison Chart")
            
            with st.spinner("Calculating Greeks across spot price range..."):
                fig_comparison = self.create_greek_comparison_chart(params)
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.info("  The gray dashed line shows the current spot price")
        
        with tab3:
            st.subheader(" ️ Delta Heatmap")
            
            with st.spinner("Generating Delta heatmap..."):
                fig_heatmap = self.create_heatmap_visualization(params)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.info("  Darker colors indicate higher Delta values")
        
        with tab4:
            st.subheader("  Advanced Analysis")
            
            st.markdown("### Mathematical Formulations")
            
            with st.expander("  Black-Scholes Greeks Formulas"):
                st.latex(r"""
                \text{Delta: } \Delta = \frac{\partial V}{\partial S} = N(d_1) \text{ (call)}, \; N(d_1) - 1 \text{ (put)}
                """)
                st.latex(r"""
                \text{Gamma: } \Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{n(d_1)}{S\sigma\sqrt{T}}
                """)
                st.latex(r"""
                \text{Theta: } \Theta = \frac{\partial V}{\partial t} = -\frac{Sn(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)
                """)
                st.latex(r"""
                \text{Vega: } \nu = \frac{\partial V}{\partial \sigma} = S n(d_1) \sqrt{T}
                """)
            
            st.markdown("### Interpretation Guide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Delta (Δ)**
                - Measures price sensitivity to underlying movement
                - Range: 0 to 1 (calls), -1 to 0 (puts)
                - Higher values = more sensitive to price changes
                """)
                
                st.markdown("""
                **Gamma (Γ)**
                - Measures rate of change of Delta
                - Always positive for long options
                - Higher values = Delta changes more rapidly
                """)
            
            with col2:
                st.markdown("""
                **Theta (Θ)**
                - Measures time decay per day
                - Usually negative for long options
                - Higher absolute values = faster time decay
                """)
                
                st.markdown("""
                **Vega (ν)**
                - Measures sensitivity to volatility changes
                - Always positive for long options
                - Higher values = more sensitive to volatility
                """)


def main():
    """Main function to run the Greeks Dashboard."""
    
    st.set_page_config(
        page_title="Greeks Dashboard - GIGA System",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
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
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    dashboard = GreeksDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()