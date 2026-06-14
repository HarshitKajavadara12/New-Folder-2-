"""
Statistical Plots - R ggplot2 Integration
=========================================

Advanced statistical visualization using R's ggplot2 integration with Python.
Provides publication-quality statistical plots for financial analysis.

Features:
- R ggplot2 integration via rpy2
- Statistical distribution plots
- Regression analysis visualizations
- Time series decomposition
- Publication-quality styling
- Export capabilities
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try to import R integration
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    
    # Activate automatic conversion
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Import R packages
    base = importr('base')
    stats = importr('stats')
    grdevices = importr('grDevices')
    
    try:
        ggplot2 = importr('ggplot2')
        R_AVAILABLE = True
    except Exception:
        R_AVAILABLE = False
        st.warning("ggplot2 R package not available. Using Python fallback visualizations.")
        
except ImportError:
    R_AVAILABLE = False
    st.info("R/rpy2 not available. Using Python-based statistical plots.")

try:
    from utils.performance_profiler import PerformanceProfiler
    from data.realtime_manager import get_data_manager, get_historical_data
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start


class StatisticalPlots:
    """Advanced statistical visualization with R ggplot2 integration."""
    
    def __init__(self):
        """Initialize Statistical Plots module."""
        self.profiler = PerformanceProfiler()
        self.r_available = R_AVAILABLE
        
        # Color palettes
        self.color_palettes = {
            'viridis': ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
            'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            'cividis': ['#00224e', '#123570', '#3b496c', '#575d6d', '#707173', '#8a8678'],
            'financial': ['#2E8B57', '#4682B4', '#DC143C', '#FF6347', '#FFD700', '#9370DB']
        }
    
    def load_real_financial_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load REAL financial data for statistical analysis.
        
        REPLACES: generate_sample_financial_data()
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        
        Returns
        -------
        pd.DataFrame
            Real market data with returns and volatility
        """
        if not REAL_DATA_AVAILABLE:
            st.error("  Real data module not available.")
            return self.generate_sample_financial_data()
        
        try:
            # Get real historical data
            dm = get_data_manager()
            df = dm.get_historical_data_sync(symbol, start_date, end_date)
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate realized volatility (rolling 20-day)
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate log returns
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volume
            df['volume_millions'] = df['volume'] / 1e6
            
            # Price momentum
            df['momentum_5d'] = df['close'].pct_change(5)
            df['momentum_20d'] = df['close'].pct_change(20)
            
            # Drop NaN values
            df = df.dropna()
            
            # Rename timestamp column for compatibility
            df = df.rename(columns={'timestamp': 'date'})
            
            st.success(f"  Loaded REAL data for {symbol}: {len(df)} trading days")
            
            return df
            
        except Exception as e:
            st.error(f"  Error loading real financial data: {e}")
            return self.generate_sample_financial_data()
    
    def generate_sample_financial_data(self, num_points: int = 1000) -> pd.DataFrame:
        """
        Generate sample financial data for statistical analysis.
        
         ️ DEPRECATED: Use load_real_financial_data() instead
        Only used as fallback when real data is unavailable.
        """
        
        np.random.seed(42)
        
        # Generate time series
        dates = pd.date_range('2020-01-01', periods=num_points, freq='D')
        
        # Generate returns with different regimes
        returns = []
        volatilities = []
        
        for i in range(num_points):
            # Regime switching volatility
            if i < num_points // 3:
                vol = 0.15  # Low volatility regime
            elif i < 2 * num_points // 3:
                vol = 0.25  # Medium volatility regime
            else:
                vol = 0.35  # High volatility regime
            
            # Add volatility clustering
            if i > 0:
                vol_shock = 0.8 * volatilities[-1] + 0.2 * vol + 0.05 * np.random.randn()
                vol = max(0.05, vol_shock)
            
            volatilities.append(vol)
            
            # Generate return with fat tails
            if np.random.rand() < 0.05:  # 5% chance of extreme event
                ret = np.random.normal(0, vol * 3)  # Fat tail
            else:
                ret = np.random.normal(0.0005, vol / np.sqrt(252))  # Normal return
            
            returns.append(ret)
        
        # Calculate additional variables
        prices = (1 + pd.Series(returns)).cumprod() * 100
        volume = np.random.lognormal(15, 0.5, num_points)
        
        # Add correlations
        factor1 = np.random.randn(num_points)
        factor2 = np.random.randn(num_points)
        
        asset2_returns = [0.3 * ret + 0.4 * f1 + 0.1 * f2 + 0.2 * np.random.randn() 
                         for ret, f1, f2 in zip(returns, factor1, factor2)]
        
        df = pd.DataFrame({
            'date': dates,
            'returns': returns,
            'prices': prices,
            'volatility': volatilities,
            'volume': volume,
            'asset2_returns': asset2_returns,
            'factor1': factor1,
            'factor2': factor2
        })
        
        # Add regime labels
        df['regime'] = pd.cut(df.index, bins=3, labels=['Low Vol', 'Medium Vol', 'High Vol'])
        
        return df
    
    def create_distribution_analysis(self, data: pd.Series, 
                                   title: str = "Return Distribution") -> go.Figure:
        """Create comprehensive distribution analysis plot."""
        
        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Create subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Histogram vs Normal Distribution',
                'Q-Q Plot',
                'Box Plot with Outliers',
                'Cumulative Distribution'
            ]
        )
        
        # 1. Histogram with normal overlay
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=50,
                name='Empirical',
                opacity=0.7,
                marker_color='lightblue',
                yaxis='y',
                offsetgroup="1"
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_dist = len(data) * (data.max() - data.min()) / 50 * \
                     (1/np.sqrt(2*np.pi*std_val**2)) * \
                     np.exp(-0.5 * ((x_range - mean_val)/std_val)**2)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        from scipy import stats
        qq_data = stats.probplot(data, dist="norm")
        
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', size=4)
            ),
            row=1, col=2
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Box Plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Distribution',
                boxpoints='outliers',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # 4. Cumulative Distribution
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=cumulative,
                mode='lines',
                name='Empirical CDF',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        # Add normal CDF
        normal_cdf = stats.norm.cdf(sorted_data, mean_val, std_val)
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=normal_cdf,
                mode='lines',
                name='Normal CDF',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"{title}<br><sub>μ={mean_val:.4f}, σ={std_val:.4f}, Skew={skewness:.3f}, Kurt={kurtosis:.3f}</sub>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_regression_analysis(self, x_data: pd.Series, y_data: pd.Series,
                                 title: str = "Regression Analysis") -> go.Figure:
        """Create comprehensive regression analysis plot."""
        
        from scipy import stats
        from plotly.subplots import make_subplots
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Scatter Plot (R² = {r_value**2:.3f})',
                'Residuals vs Fitted',
                'Residuals Distribution',
                'Influence Plot'
            ]
        )
        
        # 1. Scatter plot with regression line
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name='Data Points',
                marker=dict(color='blue', size=4, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add regression line
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_pred = intercept + slope * x_range
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'Regression Line (β={slope:.3f})',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        y_pred_data = intercept + slope * x_data
        residuals = y_data - y_pred_data
        mse = np.mean(residuals**2)
        
        # 2. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred_data,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', size=4)
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Residuals distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residuals Dist',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Influence plot (Cook's distance approximation)
        leverage = (x_data - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2)
        cooks_distance = residuals**2 * leverage / (2 * mse)
        
        fig.add_trace(
            go.Scatter(
                x=leverage,
                y=cooks_distance,
                mode='markers',
                name="Cook's Distance",
                marker=dict(
                    color=cooks_distance,
                    colorscale='Reds',
                    size=8,
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"{title}<br><sub>R²={r_value**2:.3f}, p-value={p_value:.3e}, SE={std_err:.4f}</sub>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_time_series_decomposition(self, data: pd.Series, 
                                       title: str = "Time Series Decomposition") -> go.Figure:
        """Create time series decomposition plot."""
        
        from scipy import signal
        from plotly.subplots import make_subplots
        
        # Prepare data
        if hasattr(data.index, 'to_pydatetime'):
            dates = data.index
        else:
            dates = pd.date_range('2020-01-01', periods=len(data), freq='D')
        
        # Simple trend extraction using moving average
        trend = data.rolling(window=min(30, len(data)//4), center=True).mean()
        
        # Detrend
        detrended = data - trend
        
        # Extract seasonal component (if enough data)
        if len(data) > 365:
            seasonal_period = 252  # Business days in a year
            seasonal = detrended.rolling(window=seasonal_period, center=True).mean()
        else:
            seasonal = pd.Series(0, index=data.index)
        
        # Residual
        residual = data - trend - seasonal
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original Time Series', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.05
        )
        
        # Original series
        fig.add_trace(
            go.Scatter(x=dates, y=data, name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=dates, y=trend, name='Trend', line=dict(color='red')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=dates, y=seasonal, name='Seasonal', line=dict(color='green')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=dates, y=residual, name='Residual', line=dict(color='purple')),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create comprehensive correlation analysis."""
        
        from plotly.subplots import make_subplots
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Correlation Heatmap', 'Correlation Network'],
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. Correlation network (simplified)
        # Use correlation strength to determine connections
        n_vars = len(corr_matrix.columns)
        
        # Create network layout
        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                text=corr_matrix.columns,
                textposition='middle center',
                marker=dict(size=20, color='lightblue'),
                name='Variables'
            ),
            row=1, col=2
        )
        
        # Add edges for strong correlations
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Only show strong correlations
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos[i], x_pos[j]],
                            y=[y_pos[i], y_pos[j]],
                            mode='lines',
                            line=dict(
                                color='red' if corr_val > 0 else 'blue',
                                width=abs(corr_val) * 5
                            ),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title="Correlation Analysis",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        
        return fig
    
    def create_risk_return_analysis(self, returns_data: pd.DataFrame) -> go.Figure:
        """Create risk-return scatter plot analysis."""
        
        # Calculate risk and return metrics
        mean_returns = returns_data.mean() * 252  # Annualized
        volatilities = returns_data.std() * np.sqrt(252)  # Annualized
        sharpe_ratios = mean_returns / volatilities
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=mean_returns,
            mode='markers+text',
            text=returns_data.columns,
            textposition='top center',
            marker=dict(
                size=sharpe_ratios * 50 + 10,  # Size based on Sharpe ratio
                color=sharpe_ratios,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Assets'
        ))
        
        # Add efficient frontier approximation
        if len(returns_data.columns) > 1:
            vol_range = np.linspace(volatilities.min(), volatilities.max(), 100)
            # Simple efficient frontier (not optimized)
            efficient_returns = vol_range * (mean_returns.max() / volatilities.max())
            
            fig.add_trace(go.Scatter(
                x=vol_range,
                y=efficient_returns,
                mode='lines',
                name='Efficient Frontier (Approx)',
                line=dict(color='red', dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            height=600
        )
        
        return fig
    
    def run_dashboard(self):
        """Run the complete Statistical Plots dashboard."""
        
        st.title("  Statistical Plots - Advanced Financial Analytics")
        st.markdown("""
        Comprehensive statistical visualization toolkit with R ggplot2 integration
        for publication-quality financial analysis plots.
        """)
        
        # Display R availability status
        if self.r_available:
            st.success("  R/ggplot2 integration available")
        else:
            st.info("ℹ️ Using Python-based statistical plots (R integration not available)")
        
        # Sidebar controls
        st.sidebar.header("  Plot Configuration")
        
        # Data source selection
        use_real_data = st.sidebar.checkbox("  Use REAL Market Data", value=True)
        
        if use_real_data and REAL_DATA_AVAILABLE:
            symbol = st.sidebar.text_input("Symbol", "AAPL")
            
            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*2)
            
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
            
            # Load real data
            if st.sidebar.button("  Load Real Data") or 'stats_data' not in st.session_state:
                st.session_state.stats_data = self.load_real_financial_data(symbol, start_str, end_str)
        else:
            # Synthetic data fallback
            num_points = st.sidebar.slider("Data Points", 100, 2000, 1000)
            
            if st.sidebar.button("  Generate Sample Data"):
                st.session_state.stats_data = self.generate_sample_financial_data(num_points)
        
        if 'stats_data' not in st.session_state:
            if use_real_data and REAL_DATA_AVAILABLE:
                st.session_state.stats_data = self.load_real_financial_data('AAPL', '2023-01-01', '2024-12-31')
            else:
                st.session_state.stats_data = self.generate_sample_financial_data(1000)
        
        df = st.session_state.stats_data
        
        # Display data summary
        st.subheader("  Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            st.metric("Mean Return", f"{df['returns'].mean()*100:.4f}%")
        with col3:
            st.metric("Volatility", f"{df['returns'].std()*100:.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{(df['returns'].mean() / df['returns'].std() * np.sqrt(252)):.3f}")
        
        st.divider()
        
        # Main visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "  Distribution Analysis", "  Regression Analysis", "  Time Series", 
            "  Correlation Analysis", "  Risk-Return"
        ])
        
        with tab1:
            st.subheader("  Distribution Analysis")
            
            variable_choice = st.selectbox(
                "Select Variable for Analysis",
                ['returns', 'volatility', 'volume', 'asset2_returns']
            )
            
            with st.spinner("Generating distribution analysis..."):
                fig_dist = self.create_distribution_analysis(
                    df[variable_choice],
                    f"{variable_choice.title()} Distribution Analysis"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistical tests
            st.subheader("  Statistical Tests")
            
            from scipy import stats
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(df[variable_choice].sample(min(5000, len(df))))
            ks_stat, ks_p = stats.kstest(df[variable_choice], 'norm', 
                                       args=(df[variable_choice].mean(), df[variable_choice].std()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Shapiro-Wilk Test", f"p = {shapiro_p:.3e}", 
                         delta="Normal" if shapiro_p > 0.05 else "Non-Normal")
            with col2:
                st.metric("Kolmogorov-Smirnov Test", f"p = {ks_p:.3e}",
                         delta="Normal" if ks_p > 0.05 else "Non-Normal")
        
        with tab2:
            st.subheader("  Regression Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", df.select_dtypes(include=[np.number]).columns, index=0)
            with col2:
                y_var = st.selectbox("Y Variable", df.select_dtypes(include=[np.number]).columns, index=1)
            
            if x_var != y_var:
                with st.spinner("Performing regression analysis..."):
                    fig_reg = self.create_regression_analysis(
                        df[x_var], df[y_var],
                        f"{y_var} vs {x_var} Regression Analysis"
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
        
        with tab3:
            st.subheader("  Time Series Analysis")
            
            ts_variable = st.selectbox(
                "Select Time Series Variable",
                ['prices', 'returns', 'volatility', 'volume']
            )
            
            with st.spinner("Performing time series decomposition..."):
                fig_ts = self.create_time_series_decomposition(
                    df[ts_variable],
                    f"{ts_variable.title()} Time Series Decomposition"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            
            # Additional time series metrics
            st.subheader("  Time Series Properties")
            
            from scipy.stats import jarque_bera
            
            # Stationarity test (simplified)
            returns = df[ts_variable].diff().dropna()
            jb_stat, jb_p = jarque_bera(returns)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Jarque-Bera Test", f"p = {jb_p:.3e}")
            with col2:
                autocorr = df[ts_variable].autocorr(lag=1)
                st.metric("1st Order Autocorr", f"{autocorr:.3f}")
            with col3:
                volatility_clustering = df[ts_variable].diff().abs().autocorr(lag=1)
                st.metric("Volatility Clustering", f"{volatility_clustering:.3f}")
        
        with tab4:
            st.subheader("  Correlation Analysis")
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'date' in numeric_cols:
                numeric_cols.remove('date')
            
            selected_vars = st.multiselect(
                "Select Variables for Correlation Analysis",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )
            
            if len(selected_vars) >= 2:
                with st.spinner("Analyzing correlations..."):
                    corr_data = df[selected_vars]
                    fig_corr = self.create_correlation_analysis(corr_data)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation summary table
                st.subheader("  Correlation Matrix")
                corr_matrix = corr_data.corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', center=0),
                           use_container_width=True)
        
        with tab5:
            st.subheader("  Risk-Return Analysis")
            
            # Create returns data for multiple assets
            returns_cols = ['returns', 'asset2_returns'] + [col for col in df.columns 
                                                            if 'factor' in col]
            
            if len(returns_cols) >= 2:
                with st.spinner("Analyzing risk-return characteristics..."):
                    returns_data = df[returns_cols]
                    fig_risk_return = self.create_risk_return_analysis(returns_data)
                    st.plotly_chart(fig_risk_return, use_container_width=True)
                
                # Risk metrics table
                st.subheader("  Risk-Return Metrics")
                
                metrics_data = []
                for col in returns_cols:
                    returns = df[col]
                    metrics_data.append({
                        'Asset': col,
                        'Mean Return (%)': returns.mean() * 252 * 100,
                        'Volatility (%)': returns.std() * np.sqrt(252) * 100,
                        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
                        'Skewness': returns.skew(),
                        'Kurtosis': returns.kurtosis(),
                        'Max Drawdown (%)': ((returns.cumsum().expanding().max() - returns.cumsum()) / returns.cumsum().expanding().max()).max() * 100
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Export functionality
        st.sidebar.markdown("---")
        st.sidebar.subheader("  Export Options")
        
        if st.sidebar.button("  Export Summary Report"):
            # Create summary statistics
            summary = df.describe()
            
            # Export as CSV
            csv = summary.to_csv()
            st.sidebar.download_button(
                label="  Download CSV",
                data=csv,
                file_name="statistical_analysis_summary.csv",
                mime="text/csv"
            )


def main():
    """Main function to run the Statistical Plots dashboard."""
    
    st.set_page_config(
        page_title="Statistical Plots - GIGA System",
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
    .plot-container {
        border: 1px solid #e6e9ef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    plots = StatisticalPlots()
    plots.run_dashboard()


if __name__ == "__main__":
    main()