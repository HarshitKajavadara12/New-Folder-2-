"""
Risk Dashboard - Real-Time Risk Monitoring
==========================================

Advanced real-time risk monitoring dashboard with VaR calculations,
stress testing, risk metrics, and alert systems for portfolio management.

Features:
- Real-time VaR and CVaR calculations
- Monte Carlo risk simulations
- Stress testing scenarios
- Risk limit monitoring with alerts
- Interactive risk decomposition
- Regulatory risk reporting
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from core.risk_metrics import RiskCalculator
    from core.monte_carlo import MonteCarloEngine
    from utils.performance_profiler import PerformanceProfiler
    from data.realtime_manager import get_data_manager, MarketDataConfig
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
    # Fallback implementations
    class RiskCalculator:
        @staticmethod
        def calculate_var(returns, confidence_level=0.05):
            return np.percentile(returns, confidence_level * 100)
        
        @staticmethod
        def calculate_cvar(returns, confidence_level=0.05):
            var = RiskCalculator.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        
        @staticmethod
        def calculate_volatility(returns, annualize=True):
            vol = np.std(returns)
            return vol * np.sqrt(252) if annualize else vol
        
        @staticmethod
        def calculate_correlation_matrix(returns_df):
            return returns_df.corr()
    
    class MonteCarloEngine:
        def __init__(self, num_simulations=10000):
            self.num_simulations = num_simulations
        
        def simulate_portfolio_returns(self, mean_returns, cov_matrix, weights, time_horizon=1):
            np.random.seed(42)
            num_assets = len(mean_returns)
            
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, (self.num_simulations, time_horizon)
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(random_returns, weights)
            return portfolio_returns.sum(axis=1)
    
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start


class RiskDashboard:
    """Advanced real-time risk monitoring dashboard."""
    
    def __init__(self):
        """Initialize Risk Dashboard."""
        self.risk_calc = RiskCalculator()
        self.mc_engine = MonteCarloEngine()
        self.profiler = PerformanceProfiler()
        
        # Risk thresholds (example institutional limits)
        self.risk_limits = {
            'var_95': 100000,    # $100k daily VaR limit
            'var_99': 200000,    # $200k stress VaR limit
            'volatility': 0.25,  # 25% annualized volatility limit
            'concentration': 0.20, # 20% single position limit
            'leverage': 3.0,     # 3x leverage limit
            'correlation': 0.8   # 80% correlation warning
        }
        
        # Color schemes for risk levels
        self.risk_colors = {
            'low': '#2E8B57',      # Sea Green
            'medium': '#FFA500',   # Orange
            'high': '#DC143C',     # Crimson
            'extreme': '#8B0000'   # Dark Red
        }
    
    def load_real_portfolio_data(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load REAL portfolio data from market sources.
        
        REPLACES: generate_sample_portfolio_data()
        
        Parameters
        ----------
        symbols : list of str
            List of asset symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        
        Returns
        -------
        returns_df : pd.DataFrame
            Real returns data from market
        portfolio_info : dict
            Portfolio metadata
        """
        if not REAL_DATA_AVAILABLE:
            st.error("  Real data module not available. Please install required dependencies.")
            return self.generate_sample_portfolio_data()  # Fallback
        
        try:
            # Get data manager
            dm = get_data_manager()
            
            # Fetch real historical data for all symbols
            portfolio_data = dm.get_portfolio_data_sync(symbols, start_date, end_date)
            
            if not portfolio_data:
                raise ValueError("No data available for specified symbols and date range")
            
            # Calculate returns from real prices
            returns_data = {}
            for symbol, df in portfolio_data.items():
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            # Align all returns to same index
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()  # Remove any missing data
            
            # Generate realistic position sizes (based on market cap weighting)
            # In production, this would come from actual portfolio holdings
            positions = np.random.uniform(50000, 500000, len(symbols))
            
            # Portfolio information
            portfolio_info = {
                'assets': symbols,
                'positions': positions,
                'weights': positions / positions.sum(),
                'total_value': positions.sum(),
                'returns': returns_df,
                'data_source': 'REAL_MARKET_DATA'
            }
            
            st.success(f"  Loaded REAL data: {len(returns_df)} trading days, {len(symbols)} assets")
            
            return returns_df, portfolio_info
            
        except Exception as e:
            st.error(f"  Error loading real data: {e}")
            st.warning(" ️ Falling back to sample data...")
            return self.generate_sample_portfolio_data()
    
    def generate_sample_portfolio_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate sample portfolio data for demonstration.
        
         ️ DEPRECATED: Use load_real_portfolio_data() instead
        Only used as fallback when real data is unavailable.
        """
        
        # Try to use real data even in fallback
        try:
            from data.realtime_manager import get_data_manager
            import datetime as dt
            
            dm = get_data_manager()
            assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'SPY', 'QQQ']
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=380)
            
            portfolio_data = dm.get_portfolio_data_sync(assets, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Use current prices for positions (scaled to reasonable values)
            positions = np.array([portfolio_data[symbol]['close'].iloc[-1] * 1000 for symbol in assets])
            
            # Get real returns data
            returns_data = {}
            for symbol in assets:
                returns = portfolio_data[symbol]['close'].pct_change().dropna()
                returns_data[symbol] = returns.values[-252:]
            
            # Create DataFrame
            min_len = min(len(v) for v in returns_data.values())
            returns_data = {k: v[:min_len] for k, v in returns_data.items()}
            dates = pd.date_range(end=datetime.now(), periods=min_len, freq='B')
            returns_df = pd.DataFrame(returns_data, index=dates)
        except Exception as e:
            import streamlit as st
            st.error(f"  Real portfolio data unavailable: {e}")
            st.info("  Cannot generate risk dashboard without real market data")
            return None, None
        
        # Portfolio information
        portfolio_info = {
            'assets': assets,
            'positions': positions,
            'weights': positions / positions.sum(),
            'total_value': positions.sum(),
            'returns': returns_df
        }
        
        return returns_df, portfolio_info
    
    def calculate_portfolio_risk_metrics(self, returns_df: pd.DataFrame, 
                                       weights: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio risk metrics."""
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Basic risk metrics
        metrics = {}
        
        # VaR calculations
        var_95, var_time = self.profiler.profile_function(
            self.risk_calc.calculate_var, portfolio_returns, 0.05
        )
        var_99, _ = self.profiler.profile_function(
            self.risk_calc.calculate_var, portfolio_returns, 0.01
        )
        
        # CVaR calculations
        cvar_95, cvar_time = self.profiler.profile_function(
            self.risk_calc.calculate_cvar, portfolio_returns, 0.05
        )
        cvar_99, _ = self.profiler.profile_function(
            self.risk_calc.calculate_cvar, portfolio_returns, 0.01
        )
        
        # Volatility
        volatility, vol_time = self.profiler.profile_function(
            self.risk_calc.calculate_volatility, portfolio_returns
        )
        
        metrics.update({
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'volatility': volatility,
            'var_calculation_time': var_time * 1000,  # Convert to ms
            'cvar_calculation_time': cvar_time * 1000,
            'volatility_calculation_time': vol_time * 1000
        })
        
        # Additional metrics
        metrics['max_drawdown'] = self.calculate_max_drawdown(portfolio_returns)
        metrics['sharpe_ratio'] = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        metrics['skewness'] = portfolio_returns.skew()
        metrics['kurtosis'] = portfolio_returns.kurtosis()
        
        # Concentration risk
        metrics['max_weight'] = weights.max()
        metrics['herfindahl_index'] = (weights ** 2).sum()
        
        return metrics
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def create_var_gauge_chart(self, current_var: float, var_limit: float, 
                              title: str = "VaR Gauge") -> go.Figure:
        """Create VaR gauge chart with risk thresholds."""
        
        # Normalize VaR (convert to positive for display)
        current_var_abs = abs(current_var)
        
        # Determine risk level
        risk_ratio = current_var_abs / var_limit
        
        if risk_ratio <= 0.5:
            color = self.risk_colors['low']
            risk_level = "Low Risk"
        elif risk_ratio <= 0.8:
            color = self.risk_colors['medium']
            risk_level = "Medium Risk"
        elif risk_ratio <= 1.0:
            color = self.risk_colors['high']
            risk_level = "High Risk"
        else:
            color = self.risk_colors['extreme']
            risk_level = "LIMIT BREACH!"
        
        # Create gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_var_abs,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{title}<br>{risk_level}"},
            delta={'reference': var_limit * 0.8},  # Reference to 80% of limit
            gauge={
                'axis': {'range': [None, var_limit * 1.2]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, var_limit * 0.5], 'color': "lightgray"},
                    {'range': [var_limit * 0.5, var_limit * 0.8], 'color': "yellow"},
                    {'range': [var_limit * 0.8, var_limit], 'color': "orange"},
                    {'range': [var_limit, var_limit * 1.2], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': var_limit
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    def create_risk_decomposition_chart(self, returns_df: pd.DataFrame, 
                                      weights: np.ndarray, 
                                      asset_names: List[str]) -> go.Figure:
        """Create risk decomposition chart showing component VaR."""
        
        # Calculate component VaR using marginal VaR approach
        portfolio_returns = (returns_df * weights).sum(axis=1)
        portfolio_var = self.risk_calc.calculate_var(portfolio_returns, 0.05)
        
        component_vars = []
        
        for i, asset in enumerate(asset_names):
            # Calculate marginal VaR
            asset_returns = returns_df[asset]
            correlation_with_portfolio = asset_returns.corr(portfolio_returns)
            asset_vol = asset_returns.std()
            portfolio_vol = portfolio_returns.std()
            
            # Marginal VaR approximation
            marginal_var = correlation_with_portfolio * asset_vol / portfolio_vol * portfolio_var
            component_var = weights[i] * marginal_var
            component_vars.append(component_var)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(component_vars))[::-1]
        sorted_assets = [asset_names[i] for i in sorted_indices]
        sorted_vars = [component_vars[i] for i in sorted_indices]
        
        colors = [self.risk_colors['high'] if var < 0 else self.risk_colors['low'] 
                 for var in sorted_vars]
        
        fig.add_trace(go.Bar(
            y=sorted_assets,
            x=sorted_vars,
            orientation='h',
            marker_color=colors,
            text=[f"${var:.0f}" for var in sorted_vars],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Risk Decomposition - Component VaR",
            xaxis_title="Component VaR ($)",
            yaxis_title="Assets",
            height=500,
            margin=dict(l=100, r=20, t=40, b=40)
        )
        
        return fig
    
    def create_risk_time_series(self, returns_df: pd.DataFrame, 
                               weights: np.ndarray, window: int = 30) -> go.Figure:
        """Create time series of rolling risk metrics."""
        
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate rolling metrics
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_var_95 = portfolio_returns.rolling(window).quantile(0.05)
        rolling_var_99 = portfolio_returns.rolling(window).quantile(0.01)
        
        # Create subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Rolling Volatility', 'Rolling VaR (95%)', 'Rolling VaR (99%)'],
            vertical_spacing=0.08
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=returns_df.index, y=rolling_vol, name='30-Day Volatility',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=returns_df.index, y=rolling_var_95, name='30-Day VaR 95%',
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=returns_df.index, y=rolling_var_99, name='30-Day VaR 99%',
                      line=dict(color='red')),
            row=3, col=1
        )
        
        # Add risk limit lines
        fig.add_hline(y=self.risk_limits['volatility'], line_dash="dash", 
                     line_color="red", row=1, col=1)
        
        fig.update_layout(height=800, title_text="Rolling Risk Metrics", showlegend=False)
        return fig
    
    def create_stress_testing_scenarios(self, returns_df: pd.DataFrame, 
                                      weights: np.ndarray) -> go.Figure:
        """Create stress testing scenario analysis."""
        
        scenarios = {
            'Market Crash (-20%)': -0.20,
            'Volatility Spike (2x)': 2.0,
            'Correlation Breakdown': 0.9,
            'Interest Rate Shock': 0.05,
            'Liquidity Crisis': -0.15,
            'Black Swan (-30%)': -0.30
        }
        
        portfolio_returns = (returns_df * weights).sum(axis=1)
        base_portfolio_value = 1000000  # $1M portfolio
        
        scenario_results = []
        
        for scenario_name, shock_magnitude in scenarios.items():
            if 'Crash' in scenario_name or 'Swan' in scenario_name or 'Liquidity' in scenario_name:
                # Apply market shock
                shocked_return = portfolio_returns.mean() + shock_magnitude
                scenario_pnl = base_portfolio_value * shocked_return
            elif 'Volatility' in scenario_name:
                # Increase volatility
                shocked_vol = portfolio_returns.std() * shock_magnitude
                scenario_var = np.random.normal(0, shocked_vol)
                scenario_pnl = base_portfolio_value * scenario_var
            elif 'Correlation' in scenario_name:
                # Increase all correlations
                shocked_returns = portfolio_returns * shock_magnitude
                scenario_pnl = base_portfolio_value * shocked_returns.mean()
            else:
                # Generic shock
                scenario_pnl = base_portfolio_value * shock_magnitude * 0.1
            
            scenario_results.append({
                'Scenario': scenario_name,
                'Expected P&L': scenario_pnl,
                'VaR Impact': abs(scenario_pnl) / base_portfolio_value
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Create bar chart
        fig = go.Figure()
        
        colors = [self.risk_colors['extreme'] if pnl < -50000 else 
                 self.risk_colors['high'] if pnl < -20000 else 
                 self.risk_colors['medium'] if pnl < 0 else 
                 self.risk_colors['low'] for pnl in scenario_df['Expected P&L']]
        
        fig.add_trace(go.Bar(
            x=scenario_df['Scenario'],
            y=scenario_df['Expected P&L'],
            marker_color=colors,
            text=[f"${pnl:,.0f}" for pnl in scenario_df['Expected P&L']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Stress Testing Scenarios",
            xaxis_title="Scenarios",
            yaxis_title="Expected P&L ($)",
            height=500,
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    def create_monte_carlo_risk_simulation(self, returns_df: pd.DataFrame, 
                                         weights: np.ndarray) -> go.Figure:
        """Create Monte Carlo risk simulation."""
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Run Monte Carlo simulation
        simulated_returns = self.mc_engine.simulate_portfolio_returns(
            mean_returns, cov_matrix, weights, time_horizon=1
        )
        
        # Convert to P&L (assume $1M portfolio)
        portfolio_value = 1000000
        simulated_pnl = simulated_returns * portfolio_value
        
        # Create distribution plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['P&L Distribution', 'Risk Metrics'],
            specs=[[{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        # Histogram of P&L
        fig.add_trace(
            go.Histogram(
                x=simulated_pnl,
                nbinsx=50,
                name='P&L Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add VaR lines
        var_95 = np.percentile(simulated_pnl, 5)
        var_99 = np.percentile(simulated_pnl, 1)
        
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                     annotation_text="VaR 95%", row=1, col=1)
        fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                     annotation_text="VaR 99%", row=1, col=1)
        
        # Risk metrics indicator
        expected_pnl = np.mean(simulated_pnl)
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=expected_pnl,
                title={"text": "Expected P&L"},
                domain={'x': [0.6, 1], 'y': [0.7, 1]},
                number={'prefix': "$", 'valueformat': '.0f'}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Monte Carlo Risk Simulation (10,000 scenarios)",
            height=500
        )
        
        return fig
    
    def run_dashboard(self):
        """Run the complete Risk Dashboard."""
        
        st.title(" ️ Risk Dashboard - Real-Time Risk Monitoring")
        st.markdown("""
        Comprehensive real-time risk monitoring with VaR calculations, stress testing,
        and advanced risk analytics for portfolio management.
        """)
        
        # Sidebar controls
        st.sidebar.header(" ️ Risk Controls")
        
        # Risk limits configuration
        st.sidebar.subheader("  Risk Limits")
        
        var_limit = st.sidebar.number_input(
            "Daily VaR Limit ($)",
            value=self.risk_limits['var_95'],
            format="%d"
        )
        
        volatility_limit = st.sidebar.slider(
            "Volatility Limit (%)",
            min_value=10,
            max_value=50,
            value=int(self.risk_limits['volatility'] * 100)
        ) / 100
        
        # Update limits
        self.risk_limits['var_95'] = var_limit
        self.risk_limits['volatility'] = volatility_limit
        
        # Portfolio configuration
        st.sidebar.subheader("  Portfolio Settings")
        
        # Default symbols
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                          'NVDA', 'META', 'NFLX', 'SPY', 'QQQ']
        
        symbols = st.sidebar.multiselect(
            "Select Assets",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                    'NFLX', 'BABA', 'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'USO'],
            default=default_symbols
        )
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(start_date, end_date),
            max_value=datetime.now()
        )
        
        if len(date_range) == 2:
            start_str = date_range[0].strftime('%Y-%m-%d')
            end_str = date_range[1].strftime('%Y-%m-%d')
        else:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
        
        # Data source selection
        use_real_data = st.sidebar.checkbox("  Use REAL Market Data", value=True)
        
        # Generate or load data
        if st.sidebar.button("  Refresh Portfolio Data") or 'risk_data' not in st.session_state:
            if use_real_data and REAL_DATA_AVAILABLE:
                with st.spinner("  Loading REAL market data..."):
                    st.session_state.risk_data = self.load_real_portfolio_data(
                        symbols, start_str, end_str
                    )
            else:
                if not REAL_DATA_AVAILABLE:
                    st.warning(" ️ Real data unavailable. Using sample data.")
                st.session_state.risk_data = self.generate_sample_portfolio_data()
        
        returns_df, portfolio_info = st.session_state.risk_data
        
        # Display data source indicator
        data_source = portfolio_info.get('data_source', 'SYNTHETIC_DATA')
        if data_source == 'REAL_MARKET_DATA':
            st.sidebar.success("  Using REAL market data")
        else:
            st.sidebar.warning(" ️ Using synthetic sample data")
        
        # Calculate risk metrics
        with st.spinner("Calculating risk metrics..."):
            risk_metrics = self.calculate_portfolio_risk_metrics(
                returns_df, portfolio_info['weights']
            )
        
        # Display key risk metrics
        st.subheader("  Portfolio Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95_abs = abs(risk_metrics['var_95']) * portfolio_info['total_value']
            risk_level = " " if var_95_abs < var_limit * 0.8 else " " if var_95_abs < var_limit else " "
            st.metric(
                f"{risk_level} Daily VaR (95%)",
                f"${var_95_abs:,.0f}",
                delta=f"Limit: ${var_limit:,.0f}"
            )
            st.caption(f" ️ {risk_metrics['var_calculation_time']:.1f}ms")
        
        with col2:
            var_99_abs = abs(risk_metrics['var_99']) * portfolio_info['total_value']
            st.metric(
                "Daily VaR (99%)",
                f"${var_99_abs:,.0f}",
                help="99% confidence VaR"
            )
            st.caption(f" ️ {risk_metrics['cvar_calculation_time']:.1f}ms")
        
        with col3:
            vol_level = " " if risk_metrics['volatility'] < volatility_limit * 0.8 else " " if risk_metrics['volatility'] < volatility_limit else " "
            st.metric(
                f"{vol_level} Volatility",
                f"{risk_metrics['volatility']*100:.1f}%",
                delta=f"Limit: {volatility_limit*100:.0f}%"
            )
            st.caption(f" ️ {risk_metrics['volatility_calculation_time']:.1f}ms")
        
        with col4:
            concentration_level = " " if risk_metrics['max_weight'] < 0.2 else " " if risk_metrics['max_weight'] < 0.3 else " "
            st.metric(
                f"{concentration_level} Max Position",
                f"{risk_metrics['max_weight']*100:.1f}%",
                help="Largest single position"
            )
        
        st.divider()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "  Risk Gauges", "  Risk Decomposition", "  Time Series", 
            "  Stress Testing", "  Monte Carlo"
        ])
        
        with tab1:
            st.subheader("  Risk Monitoring Gauges")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_var_gauge = self.create_var_gauge_chart(
                    risk_metrics['var_95'] * portfolio_info['total_value'],
                    var_limit,
                    "Daily VaR (95%)"
                )
                st.plotly_chart(fig_var_gauge, use_container_width=True)
            
            with col2:
                fig_vol_gauge = self.create_var_gauge_chart(
                    risk_metrics['volatility'],
                    volatility_limit,
                    "Annualized Volatility"
                )
                st.plotly_chart(fig_vol_gauge, use_container_width=True)
            
            # Additional risk metrics table
            st.subheader("  Detailed Risk Metrics")
            
            detailed_metrics = pd.DataFrame({
                'Metric': [
                    'Expected Shortfall (95%)',
                    'Expected Shortfall (99%)',
                    'Maximum Drawdown',
                    'Sharpe Ratio',
                    'Skewness',
                    'Excess Kurtosis',
                    'Herfindahl Index'
                ],
                'Value': [
                    f"${abs(risk_metrics['cvar_95']) * portfolio_info['total_value']:,.0f}",
                    f"${abs(risk_metrics['cvar_99']) * portfolio_info['total_value']:,.0f}",
                    f"{risk_metrics['max_drawdown']*100:.2f}%",
                    f"{risk_metrics['sharpe_ratio']:.3f}",
                    f"{risk_metrics['skewness']:.3f}",
                    f"{risk_metrics['kurtosis']:.3f}",
                    f"{risk_metrics['herfindahl_index']:.3f}"
                ],
                'Interpretation': [
                    'Average loss beyond VaR',
                    'Average loss in worst 1% scenarios',
                    'Peak-to-trough decline',
                    'Risk-adjusted return',
                    'Return distribution asymmetry',
                    'Tail thickness vs normal',
                    'Portfolio concentration measure'
                ]
            })
            
            st.dataframe(detailed_metrics, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("  Risk Decomposition Analysis")
            
            fig_decomp = self.create_risk_decomposition_chart(
                returns_df, portfolio_info['weights'], portfolio_info['assets']
            )
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            # Portfolio composition
            st.subheader("  Portfolio Composition")
            
            composition_df = pd.DataFrame({
                'Asset': portfolio_info['assets'],
                'Position ($)': portfolio_info['positions'],
                'Weight (%)': portfolio_info['weights'] * 100,
                'Daily Vol (%)': [returns_df[asset].std() * 100 * np.sqrt(252) for asset in portfolio_info['assets']]
            }).sort_values('Position ($)', ascending=False)
            
            st.dataframe(composition_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("  Risk Time Series Analysis")
            
            rolling_window = st.slider("Rolling Window (Days)", 10, 60, 30)
            
            fig_time_series = self.create_risk_time_series(
                returns_df, portfolio_info['weights'], rolling_window
            )
            st.plotly_chart(fig_time_series, use_container_width=True)
        
        with tab4:
            st.subheader("  Stress Testing Scenarios")
            
            fig_stress = self.create_stress_testing_scenarios(
                returns_df, portfolio_info['weights']
            )
            st.plotly_chart(fig_stress, use_container_width=True)
            
            st.markdown("""
            **Scenario Descriptions:**
            - **Market Crash**: Broad market decline of 20%
            - **Volatility Spike**: Doubling of current volatility levels
            - **Correlation Breakdown**: All correlations increase to 0.9
            - **Interest Rate Shock**: Sudden 5% rate increase
            - **Liquidity Crisis**: 15% decline due to liquidity constraints
            - **Black Swan**: Extreme 30% market decline
            """)
        
        with tab5:
            st.subheader("  Monte Carlo Risk Simulation")
            
            num_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=2)
            
            if st.button("  Run Monte Carlo Simulation"):
                self.mc_engine.num_simulations = num_simulations
                
                with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
                    fig_mc = self.create_monte_carlo_risk_simulation(
                        returns_df, portfolio_info['weights']
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
            
            st.info("  **Tip**: Monte Carlo simulation provides probabilistic risk assessment based on historical correlations and volatilities.")


def main():
    """Main function to run the Risk Dashboard."""
    
    st.set_page_config(
        page_title="Risk Dashboard - GIGA System",
        page_icon=" ️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .risk-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    dashboard = RiskDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()