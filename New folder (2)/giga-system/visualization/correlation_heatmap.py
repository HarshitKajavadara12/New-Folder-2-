"""
Correlation Heatmap - Asset Correlation Matrices
===============================================

Advanced correlation analysis and visualization with interactive heatmaps,
hierarchical clustering, and network analysis for portfolio management.

Features:
- Interactive correlation heatmaps
- Hierarchical clustering of correlations
- Time-varying correlation analysis
- Network-based correlation visualization
- Statistical significance testing
- Factor-based correlation decomposition
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
    from utils.performance_profiler import PerformanceProfiler
    from data.market_data import DataProvider
    from data.realtime_manager import get_data_manager, get_correlation_matrix
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False
    # Fallback implementations
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start
    
    class DataProvider:
        @staticmethod
        def get_sample_data(assets, start_date, end_date):
            np.random.seed(42)
            dates = pd.date_range(start_date, end_date, freq='B')
            data = {}
            
            for i, asset in enumerate(assets):
                # Generate correlated returns
                base_return = np.random.normal(0.0005, 0.02, len(dates))
                if i > 0:
                    correlation = 0.3 + 0.4 * np.random.random()
                    base_return += correlation * data[assets[0]]
                
                prices = (1 + pd.Series(base_return)).cumprod() * 100
                data[asset] = prices
            
            return pd.DataFrame(data, index=dates)


class CorrelationHeatmap:
    """Advanced correlation analysis and visualization dashboard."""
    
    def __init__(self):
        """Initialize Correlation Heatmap dashboard."""
        self.profiler = PerformanceProfiler()
        self.data_provider = DataProvider()
        
        # Color schemes for different correlation visualizations
        self.color_schemes = {
            'diverging': 'RdBu_r',
            'sequential': 'Viridis',
            'cool_warm': 'coolwarm',
            'plasma': 'Plasma',
            'custom_financial': [[0, '#8B0000'], [0.25, '#DC143C'], [0.5, '#FFFFFF'], 
                               [0.75, '#4169E1'], [1, '#000080']]
        }
        
        # Default asset universe
        self.default_assets = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'BAC', 'WFC', 'C', 'MS', 'GS',
            'SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM',
            'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'LQD'
        ]
    
    def load_real_correlation_data(self, symbols: List[str], 
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load REAL correlation data from market sources.
        
        REPLACES: generate_sample_correlation_data()
        
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
        pd.DataFrame
            Real returns data for correlation calculation
        """
        if not REAL_DATA_AVAILABLE:
            st.error("  Real data module not available.")
            st.info("  Check realtime_manager is properly installed")
            return None
        
        try:
            # Get real portfolio data
            dm = get_data_manager()
            portfolio_data = dm.get_portfolio_data_sync(symbols, start_date, end_date)
            
            if not portfolio_data:
                raise ValueError("No data available for specified symbols")
            
            # Calculate returns for each asset
            returns_data = {}
            for symbol, df in portfolio_data.items():
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            # Align all returns to same index
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            st.success(f"  Loaded REAL correlation data: {len(returns_df)} days, {len(symbols)} assets")
            
            return returns_df
            
        except Exception as e:
            st.error(f"  Error loading real correlation data: {e}")
            st.info("  Check your market data connection")
            return None
    
    def generate_sample_correlation_data(self, num_assets: int = 20, 
                                       num_days: int = 252) -> pd.DataFrame:
        """
        Generate sample correlation data with realistic market structure.
        
         ️ DEPRECATED: Use load_real_correlation_data() instead
        Only used as fallback when real data is unavailable.
        """
        
        np.random.seed(42)
        
        # Select assets
        assets = self.default_assets[:num_assets]
        
        # Generate factor-based returns structure
        # Factor 1: Market factor
        market_factor = np.random.normal(0, 0.02, num_days)
        
        # Factor 2: Sector factors
        tech_factor = np.random.normal(0, 0.015, num_days)
        finance_factor = np.random.normal(0, 0.015, num_days)
        
        # Factor 3: Size factor
        size_factor = np.random.normal(0, 0.01, num_days)
        
        # Generate returns with factor structure
        returns_data = {}
        
        for i, asset in enumerate(assets):
            # Determine sector and characteristics
            if any(x in asset for x in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']):
                # Tech stocks
                sector_exposure = 0.6
                sector_factor_return = tech_factor
            elif any(x in asset for x in ['JPM', 'BAC', 'WFC', 'C', 'MS', 'GS']):
                # Financial stocks
                sector_exposure = 0.5
                sector_factor_return = finance_factor
            else:
                # ETFs and others
                sector_exposure = 0.3
                sector_factor_return = np.random.normal(0, 0.01, num_days)
            
            # Market exposure (beta)
            market_beta = 0.8 + 0.6 * np.random.random()
            
            # Size exposure
            size_beta = np.random.normal(0, 0.3)
            
            # Idiosyncratic risk
            idiosyncratic = np.random.normal(0, 0.015, num_days)
            
            # Combine factors
            asset_returns = (market_beta * market_factor + 
                           sector_exposure * sector_factor_return + 
                           size_beta * size_factor + 
                           idiosyncratic)
            
            returns_data[asset] = asset_returns
        
        # Create DataFrame
        dates = pd.date_range(end=datetime.now(), periods=num_days, freq='B')
        df = pd.DataFrame(returns_data, index=dates)
        
        return df
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame, 
                                   method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix with different methods."""
        
        if method == 'pearson':
            return returns_data.corr(method='pearson')
        elif method == 'spearman':
            return returns_data.corr(method='spearman')
        elif method == 'kendall':
            return returns_data.corr(method='kendall')
        else:
            return returns_data.corr()
    
    def calculate_rolling_correlations(self, returns_data: pd.DataFrame, 
                                     window: int = 60,
                                     asset1: str = None, asset2: str = None) -> pd.DataFrame:
        """Calculate rolling correlations between assets."""
        
        if asset1 is None:
            asset1 = returns_data.columns[0]
        if asset2 is None:
            asset2 = returns_data.columns[1]
        
        # Rolling correlation
        rolling_corr = returns_data[asset1].rolling(window=window).corr(returns_data[asset2])
        
        # Rolling correlations with all assets
        rolling_corr_matrix = {}
        for asset in returns_data.columns:
            rolling_corr_matrix[asset] = returns_data[asset1].rolling(window=window).corr(returns_data[asset])
        
        return pd.DataFrame(rolling_corr_matrix, index=returns_data.index)
    
    def perform_hierarchical_clustering(self, corr_matrix: pd.DataFrame) -> Tuple[List, np.ndarray]:
        """Perform hierarchical clustering on correlation matrix."""
        
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform linkage
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # Get dendrogram
        dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
        
        # Get clustered order
        clustered_order = dendro['leaves']
        clustered_labels = [corr_matrix.columns[i] for i in clustered_order]
        
        return clustered_labels, linkage_matrix
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                 title: str = "Correlation Heatmap",
                                 color_scheme: str = 'RdBu_r',
                                 cluster: bool = False) -> go.Figure:
        """Create interactive correlation heatmap."""
        
        # Perform clustering if requested
        if cluster and len(corr_matrix) > 2:
            try:
                clustered_labels, _ = self.perform_hierarchical_clustering(corr_matrix)
                plot_matrix = corr_matrix.reindex(clustered_labels, columns=clustered_labels)
            except:
                plot_matrix = corr_matrix
                clustered_labels = corr_matrix.columns.tolist()
        else:
            plot_matrix = corr_matrix
            clustered_labels = corr_matrix.columns.tolist()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=plot_matrix.values,
            x=clustered_labels,
            y=clustered_labels,
            colorscale=color_scheme,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(plot_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.2
            ),
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Assets",
            yaxis_title="Assets",
            height=600,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0}
        )
        
        return fig
    
    def create_correlation_network(self, corr_matrix: pd.DataFrame,
                                 threshold: float = 0.5) -> go.Figure:
        """Create network visualization of correlations."""
        
        import networkx as nx
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for asset in corr_matrix.columns:
            G.add_node(asset)
        
        # Add edges for correlations above threshold
        for i, asset1 in enumerate(corr_matrix.columns):
            for j, asset2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        G.add_edge(asset1, asset2, weight=abs(corr_val), correlation=corr_val)
        
        # Calculate layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            # Fallback to circular layout
            pos = nx.circular_layout(G)
        
        # Extract node and edge data
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge information
            correlation = G[edge[0]][edge[1]]['correlation']
            edge_info.append(f"{edge[0]}-{edge[1]}: {correlation:.3f}")
        
        # Add edge traces
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add node traces
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='middle center',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hoverinfo='text',
            hovertext=node_text,
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Correlation Network (threshold = {threshold})",
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(b=20,l=5,r=5,t=40)
        )
        
        return fig
    
    def create_time_varying_correlation(self, returns_data: pd.DataFrame,
                                      asset1: str, asset2: str,
                                      window: int = 60) -> go.Figure:
        """Create time-varying correlation plot."""
        
        # Calculate rolling correlation
        rolling_corr = returns_data[asset1].rolling(window=window).corr(returns_data[asset2])
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'Rolling Correlation ({window}-day window)', 'Price Series'],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1
        )
        
        # Rolling correlation
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=rolling_corr,
                mode='lines',
                name=f'{asset1}-{asset2} Correlation',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add correlation bands
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                     annotation_text="High Correlation", row=1, col=1)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", 
                     annotation_text="Negative Correlation", row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
        
        # Price series (normalized)
        price1_norm = (returns_data[asset1].cumsum() - returns_data[asset1].cumsum().mean()) / returns_data[asset1].cumsum().std()
        price2_norm = (returns_data[asset2].cumsum() - returns_data[asset2].cumsum().mean()) / returns_data[asset2].cumsum().std()
        
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=price1_norm,
                mode='lines',
                name=f'{asset1} (normalized)',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=price2_norm,
                mode='lines',
                name=f'{asset2} (normalized)',
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Time-Varying Correlation: {asset1} vs {asset2}",
            height=800,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Returns", row=2, col=1)
        
        return fig
    
    def create_correlation_distribution(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation distribution analysis."""
        
        # Extract upper triangular correlations (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.values[mask]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Correlation Distribution', 'Box Plot']
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=correlations,
                nbinsx=30,
                name='Correlation Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=correlations,
                name='Correlation Range',
                boxpoints='outliers',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Add statistics
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        fig.add_vline(x=mean_corr, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_corr:.3f}", row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Correlation Distribution Analysis (n={len(correlations)} pairs)",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        return fig
    
    def calculate_correlation_statistics(self, corr_matrix: pd.DataFrame) -> Dict:
        """Calculate comprehensive correlation statistics."""
        
        # Extract unique correlations
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.values[mask]
        
        stats = {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'median_correlation': np.median(correlations),
            'high_correlation_pairs': np.sum(np.abs(correlations) > 0.7),
            'negative_correlation_pairs': np.sum(correlations < -0.3),
            'total_pairs': len(correlations)
        }
        
        # Most correlated pairs
        corr_abs = np.abs(corr_matrix)
        np.fill_diagonal(corr_abs.values, 0)  # Remove diagonal
        
        # Find top correlations
        top_indices = np.unravel_index(np.argpartition(-corr_abs.values, -5, axis=None)[-5:], 
                                     corr_abs.shape)
        
        top_pairs = []
        for i, j in zip(top_indices[0], top_indices[1]):
            if i < j:  # Avoid duplicates
                top_pairs.append({
                    'asset1': corr_matrix.columns[i],
                    'asset2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        stats['top_correlated_pairs'] = sorted(top_pairs, 
                                             key=lambda x: abs(x['correlation']), 
                                             reverse=True)
        
        return stats
    
    def run_dashboard(self):
        """Run the complete Correlation Heatmap dashboard."""
        
        st.title("  Correlation Heatmap - Asset Correlation Analysis")
        st.markdown("""
        Advanced correlation analysis with interactive heatmaps, hierarchical clustering,
        and network visualization for portfolio management and risk analysis.
        """)
        
        # Sidebar controls
        st.sidebar.header("  Configuration")
        
        # Data source selection
        use_real_data = st.sidebar.checkbox("  Use REAL Market Data", value=True)
        
        if use_real_data and REAL_DATA_AVAILABLE:
            # Asset selection
            available_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
                              'JPM', 'BAC', 'WFC', 'C', 'MS', 'GS',
                              'SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM',
                              'GLD', 'SLV', 'USO', 'TLT', 'HYG', 'LQD']
            
            selected_assets = st.sidebar.multiselect(
                "Select Assets",
                available_assets,
                default=available_assets[:10]
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
            
            # Load real data
            if st.sidebar.button("  Load Real Data") or 'corr_data' not in st.session_state:
                if selected_assets:
                    st.session_state.corr_data = self.load_real_correlation_data(selected_assets, start_str, end_str)
                else:
                    st.error("  Please select at least 2 assets for correlation analysis")
                    st.info("  Use the multiselect above to choose assets")
                    return
        else:
            st.error("  Synthetic data mode not available")
            st.info("  Please enable 'Use Real Market Data' option above")
            return
        
        if 'corr_data' not in st.session_state:
            if use_real_data and REAL_DATA_AVAILABLE:
                default_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'SPY', 'QQQ']
                st.session_state.corr_data = self.load_real_correlation_data(default_assets, '2024-01-01', '2024-12-31')
            else:
                st.error("  No correlation data loaded")
                st.info("  Click 'Load Real Data' button to fetch market data")
                return
        
        returns_data = st.session_state.corr_data
        
        # Correlation method
        correlation_method = st.sidebar.selectbox(
            "Correlation Method",
            ['pearson', 'spearman', 'kendall'],
            help="Pearson: linear relationships, Spearman: monotonic relationships, Kendall: robust to outliers"
        )
        
        # Visualization options
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            ['RdBu_r', 'coolwarm', 'Viridis', 'Plasma'],
            help="Color scheme for correlation visualization"
        )
        
        # Calculate correlation matrix
        with st.spinner("Calculating correlation matrix..."):
            corr_matrix, calc_time = self.profiler.profile_function(
                self.calculate_correlation_matrix, returns_data, correlation_method
            )
        
        # Calculate statistics
        corr_stats = self.calculate_correlation_statistics(corr_matrix)
        
        # Display overview metrics
        st.subheader("  Correlation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Correlation", f"{corr_stats['mean_correlation']:.3f}")
        with col2:
            st.metric("Std Deviation", f"{corr_stats['std_correlation']:.3f}")
        with col3:
            st.metric("High Corr Pairs", f"{corr_stats['high_correlation_pairs']}")
        with col4:
            st.metric("Calculation Time", f"{calc_time*1000:.1f}ms")
        
        st.divider()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            " ️ Heatmap", " ️ Network", "  Time-Varying", "  Distribution", "  Statistics"
        ])
        
        with tab1:
            st.subheader(" ️ Interactive Correlation Heatmap")
            
            # Heatmap options
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_option = st.checkbox("Apply Hierarchical Clustering", value=False,
                                           help="Group similar assets together")
            with col2:
                show_values = st.checkbox("Show Correlation Values", value=True)
            
            # Create heatmap
            with st.spinner("Generating correlation heatmap..."):
                fig_heatmap = self.create_correlation_heatmap(
                    corr_matrix, 
                    f"Asset Correlation Matrix ({correlation_method.title()})",
                    color_scheme,
                    cluster_option
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Correlation summary table
            st.subheader("  Top Correlated Pairs")
            
            if corr_stats['top_correlated_pairs']:
                top_pairs_df = pd.DataFrame(corr_stats['top_correlated_pairs'])
                top_pairs_df['correlation'] = top_pairs_df['correlation'].round(4)
                st.dataframe(top_pairs_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader(" ️ Correlation Network Analysis")
            
            # Network threshold
            threshold = st.slider(
                "Correlation Threshold", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5, 
                step=0.1,
                help="Only show correlations above this threshold"
            )
            
            # Create network visualization
            with st.spinner("Building correlation network..."):
                try:
                    fig_network = self.create_correlation_network(corr_matrix, threshold)
                    st.plotly_chart(fig_network, use_container_width=True)
                except Exception as e:
                    st.error(f"Network visualization not available: {e}")
                    st.info("Install networkx for network visualization: pip install networkx")
            
            # Network statistics
            st.subheader("  Network Statistics")
            
            # Count connections
            connections = 0
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) >= threshold:
                        connections += 1
            
            density = connections / (len(corr_matrix) * (len(corr_matrix) - 1) / 2)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Connections", connections)
            with col2:
                st.metric("Network Density", f"{density:.2%}")
            with col3:
                st.metric("Isolated Assets", 
                         len(corr_matrix) - len([i for i in range(len(corr_matrix)) 
                                               if any(abs(corr_matrix.iloc[i, j]) >= threshold 
                                                     for j in range(len(corr_matrix)) if i != j)]))
        
        with tab3:
            st.subheader("  Time-Varying Correlations")
            
            # Asset selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                asset1 = st.selectbox("First Asset", returns_data.columns, index=0)
            with col2:
                asset2 = st.selectbox("Second Asset", returns_data.columns, index=1)
            with col3:
                rolling_window = st.slider("Rolling Window (Days)", 20, 120, 60)
            
            if asset1 != asset2:
                # Create time-varying correlation plot
                with st.spinner("Calculating time-varying correlations..."):
                    fig_time_varying = self.create_time_varying_correlation(
                        returns_data, asset1, asset2, rolling_window
                    )
                    st.plotly_chart(fig_time_varying, use_container_width=True)
                
                # Correlation statistics over time
                rolling_corr = returns_data[asset1].rolling(window=rolling_window).corr(returns_data[asset2])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Rolling Correlation", f"{rolling_corr.mean():.3f}")
                with col2:
                    st.metric("Correlation Volatility", f"{rolling_corr.std():.3f}")
                with col3:
                    st.metric("Current Correlation", f"{rolling_corr.iloc[-1]:.3f}")
        
        with tab4:
            st.subheader("  Correlation Distribution Analysis")
            
            # Create distribution plot
            with st.spinner("Analyzing correlation distribution..."):
                fig_distribution = self.create_correlation_distribution(corr_matrix)
                st.plotly_chart(fig_distribution, use_container_width=True)
            
            # Distribution statistics
            st.subheader("  Distribution Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean", f"{corr_stats['mean_correlation']:.4f}")
                st.metric("Standard Deviation", f"{corr_stats['std_correlation']:.4f}")
                st.metric("Median", f"{corr_stats['median_correlation']:.4f}")
            
            with col2:
                st.metric("Minimum", f"{corr_stats['min_correlation']:.4f}")
                st.metric("Maximum", f"{corr_stats['max_correlation']:.4f}")
                st.metric("Range", f"{corr_stats['max_correlation'] - corr_stats['min_correlation']:.4f}")
        
        with tab5:
            st.subheader("  Detailed Correlation Statistics")
            
            # Summary statistics
            summary_stats = {
                'Metric': [
                    'Total Asset Pairs',
                    'Mean Correlation',
                    'Standard Deviation',
                    'High Correlation Pairs (|r| > 0.7)',
                    'Moderate Correlation Pairs (0.3 < |r| < 0.7)',
                    'Low Correlation Pairs (|r| < 0.3)',
                    'Negative Correlation Pairs (r < -0.3)',
                    'Maximum Correlation',
                    'Minimum Correlation'
                ],
                'Value': [
                    corr_stats['total_pairs'],
                    f"{corr_stats['mean_correlation']:.4f}",
                    f"{corr_stats['std_correlation']:.4f}",
                    corr_stats['high_correlation_pairs'],
                    np.sum((np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]) > 0.3) & 
                          (np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]) < 0.7)),
                    np.sum(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]) < 0.3),
                    corr_stats['negative_correlation_pairs'],
                    f"{corr_stats['max_correlation']:.4f}",
                    f"{corr_stats['min_correlation']:.4f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_stats)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Export options
            st.subheader("  Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download correlation matrix
                csv_corr = corr_matrix.to_csv()
                st.download_button(
                    "  Download Correlation Matrix",
                    csv_corr,
                    "correlation_matrix.csv",
                    "text/csv"
                )
            
            with col2:
                # Download returns data
                csv_returns = returns_data.to_csv()
                st.download_button(
                    "  Download Returns Data",
                    csv_returns,
                    "returns_data.csv",
                    "text/csv"
                )


def main():
    """Main function to run the Correlation Heatmap dashboard."""
    
    st.set_page_config(
        page_title="Correlation Heatmap - GIGA System",
        page_icon=" ",
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
    .correlation-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .correlation-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .correlation-low {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    dashboard = CorrelationHeatmap()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()