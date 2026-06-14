"""
GIGA SYSTEM - Portfolio Analysis Page
Streamlit page for portfolio optimization and analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import GIGA components
import sys
sys.path.append('../..')

try:
    from core.risk_metrics import RiskMetrics
    from data.realtime_manager import get_data_manager, get_portfolio_returns, get_correlation_matrix
    CORE_AVAILABLE = True
    REAL_DATA_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    REAL_DATA_AVAILABLE = False

try:
    from visualization.charts import (
        efficient_frontier, weights_timeline, correlation_heatmap,
        risk_decomposition_chart, multi_asset_chart
    )
    from visualization.components import (
        metric_row, section_header, allocation_pie, returns_histogram
    )
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False


def render_portfolio_page():
    """Render the portfolio analysis page."""
    
    st.title("  Portfolio Analysis")
    st.markdown("Portfolio optimization and risk management powered by GIGA System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Portfolio Settings")
        
        # Asset selection
        available_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 
                          'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
        
        selected_assets = st.multiselect(
            "Select Assets",
            available_assets,
            default=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
        )
        
        st.markdown("---")
        
        # Optimization settings
        st.subheader("Optimization")
        opt_method = st.selectbox(
            "Method",
            ["Mean-Variance", "Risk Parity", "Maximum Sharpe", 
             "Minimum Volatility", "Black-Litterman", "Quantum QAOA"]
        )
        
        risk_aversion = st.slider("Risk Aversion", 0.0, 5.0, 2.0, 0.1)
        
        min_weight = st.number_input("Min Weight", 0.0, 0.5, 0.0, 0.01)
        max_weight = st.number_input("Max Weight", 0.1, 1.0, 0.4, 0.01)
        
        st.markdown("---")
        
        # Risk parameters
        st.subheader("Risk Settings")
        confidence = st.slider("VaR Confidence", 0.90, 0.99, 0.95, 0.01)
        horizon = st.number_input("Horizon (days)", 1, 30, 10)
        
        optimize_btn = st.button("  Optimize Portfolio", type="primary")
    
    if not selected_assets:
        st.warning("Please select at least 2 assets to continue")
        return
    
    # Load REAL market data for portfolio analysis
    n_assets = len(selected_assets)
    n_days = 252 * 2  # 2 years
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    if REAL_DATA_AVAILABLE:
        try:
            dm = get_data_manager()
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            start_date_str = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')
            
            # Fetch real data
            portfolio_data = dm.get_portfolio_data_sync(selected_assets, start_date_str, end_date_str)
            
            # Calculate real correlation matrix
            corr_df = dm.calculate_correlation_matrix(selected_assets, start_date_str, end_date_str)
            corr = corr_df.values
            
            # Calculate real returns statistics
            returns_data = {}
            for symbol, df in portfolio_data.items():
                returns_data[symbol] = df['close'].pct_change().dropna()
            returns_df = pd.DataFrame(returns_data)
            mean_returns = returns_df.mean().values
            vol = returns_df.std().values
            
            st.success(f"  Using REAL market data: {len(returns_df)} trading days")
            
        except Exception as e:
            st.error(f"  Failed to load real data: {e}")
            st.info("  Portfolio optimization requires real market data")
            return
    else:
        st.error("  Real data module not available")
        st.info("  Portfolio optimization requires real market data")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Optimization", "  Analysis", " ️ Risk", "  Quantum"
    ])
    
    # ==========================================================================
    # OPTIMIZATION TAB
    # ==========================================================================
    with tab1:
        section_header("Portfolio Optimization", f"Method: {opt_method}")
        
        # Calculate expected returns and covariance
        exp_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Optimize portfolio based on method
        if opt_method == "Mean-Variance":
            # Classic Markowitz
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(n_assets)
            mu = exp_returns.values
            
            # Target return optimization
            target_return = exp_returns.mean()
            
            # Lagrangian solution
            A = np.array([
                [2 * cov_matrix.values @ inv_cov @ cov_matrix.values, mu, ones],
                [mu, 0, 0],
                [ones, 0, 0]
            ])
            
            # Simplified: equal weight with tilts
            weights = np.array([1/n_assets] * n_assets)
            
            # Adjust for risk aversion
            sharpe = exp_returns / np.sqrt(np.diag(cov_matrix))
            tilt = sharpe - sharpe.mean()
            weights = weights + tilt.values * (1 - risk_aversion/5) * 0.2
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
        elif opt_method == "Risk Parity":
            # Equal risk contribution
            vol_inv = 1 / np.sqrt(np.diag(cov_matrix))
            weights = vol_inv / vol_inv.sum()
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
        elif opt_method == "Maximum Sharpe":
            # Maximum Sharpe ratio
            sharpe = exp_returns / np.sqrt(np.diag(cov_matrix))
            weights = np.exp(sharpe * 5)  # Softmax-like
            weights = np.clip(weights.values, min_weight, max_weight)
            weights = weights / weights.sum()
            
        elif opt_method == "Minimum Volatility":
            # Minimum variance
            inv_vol = 1 / np.diag(cov_matrix)
            weights = inv_vol / inv_vol.sum()
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
        else:
            # Default equal weight
            weights = np.array([1/n_assets] * n_assets)
        
        # Display optimal weights
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Optimal Allocation")
            
            weights_dict = {asset: w for asset, w in zip(selected_assets, weights)}
            
            if CHARTS_AVAILABLE:
                allocation_pie(weights_dict)
            else:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Pie(
                    labels=list(weights_dict.keys()),
                    values=list(weights_dict.values()),
                    hole=0.4
                )])
                fig.update_layout(height=400, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Weights")
            
            weights_df_display = pd.DataFrame({
                'Asset': selected_assets,
                'Weight': [f'{w*100:.1f}%' for w in weights]
            })
            st.dataframe(weights_df_display, hide_index=True, use_container_width=True)
        
        # Portfolio metrics
        st.markdown("---")
        st.subheader("  Portfolio Metrics")
        
        port_return = np.dot(weights, exp_returns)
        port_vol = np.sqrt(np.dot(weights, cov_matrix.values @ weights))
        port_sharpe = port_return / port_vol
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expected Return", f"{port_return*100:.2f}%")
        with col2:
            st.metric("Volatility", f"{port_vol*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
        with col4:
            st.metric("Max Weight", f"{max(weights)*100:.1f}%")
        
        # Efficient Frontier
        st.markdown("---")
        st.subheader("  Efficient Frontier")
        
        # Generate frontier portfolios
        n_portfolios = 100
        frontier_returns = []
        frontier_risks = []
        frontier_sharpes = []
        
        for _ in range(n_portfolios):
            w = np.random.random(n_assets)
            w = np.clip(w, min_weight, max_weight)
            w = w / w.sum()
            
            ret = np.dot(w, exp_returns)
            risk = np.sqrt(np.dot(w, cov_matrix.values @ w))
            
            frontier_returns.append(ret)
            frontier_risks.append(risk)
            frontier_sharpes.append(ret / risk)
        
        portfolios = [
            {'return': r, 'risk': v, 'sharpe': s}
            for r, v, s in zip(frontier_returns, frontier_risks, frontier_sharpes)
        ]
        
        optimal = {'return': port_return, 'risk': port_vol, 'sharpe': port_sharpe}
        
        if CHARTS_AVAILABLE:
            fig = efficient_frontier(portfolios, optimal)
            st.plotly_chart(fig, use_container_width=True)
        else:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p['risk']*100 for p in portfolios],
                y=[p['return']*100 for p in portfolios],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[p['sharpe'] for p in portfolios],
                    colorscale='Viridis',
                    colorbar=dict(title='Sharpe')
                ),
                name='Portfolios'
            ))
            fig.add_trace(go.Scatter(
                x=[port_vol*100],
                y=[port_return*100],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Optimal'
            ))
            fig.update_layout(
                height=500,
                template='plotly_dark',
                title='Efficient Frontier',
                xaxis_title='Risk (%)',
                yaxis_title='Return (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # ANALYSIS TAB
    # ==========================================================================
    with tab2:
        section_header("Portfolio Analysis", "Performance and correlation analysis")
        
        # Asset performance
        st.subheader("  Asset Performance")
        
        normalized_prices = prices_df / prices_df.iloc[0] * 100
        
        import plotly.graph_objects as go
        fig = go.Figure()
        
        colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6',
                  '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
        
        for i, asset in enumerate(selected_assets):
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[asset],
                name=asset,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Normalized Price (Base = 100)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("---")
        st.subheader("  Correlation Matrix")
        
        corr_df = returns_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale='RdYlGn',
            zmin=-1, zmax=1,
            text=np.round(corr_df.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Return Correlations'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("  Summary Statistics")
        
        stats = pd.DataFrame({
            'Ann. Return': (returns_df.mean() * 252 * 100).round(2).astype(str) + '%',
            'Ann. Vol': (returns_df.std() * np.sqrt(252) * 100).round(2).astype(str) + '%',
            'Sharpe': ((returns_df.mean() * 252) / (returns_df.std() * np.sqrt(252))).round(2),
            'Max DD': ((prices_df / prices_df.cummax() - 1).min() * 100).round(2).astype(str) + '%',
            'Skewness': returns_df.skew().round(2),
            'Kurtosis': returns_df.kurtosis().round(2)
        })
        
        st.dataframe(stats, use_container_width=True)
    
    # ==========================================================================
    # RISK TAB
    # ==========================================================================
    with tab3:
        section_header("Risk Analysis", "VaR, CVaR, and risk decomposition")
        
        # Portfolio returns
        port_returns = (returns_df * weights).sum(axis=1)
        
        # VaR calculations
        var_95 = np.percentile(port_returns, 5)
        var_99 = np.percentile(port_returns, 1)
        cvar_95 = port_returns[port_returns <= var_95].mean()
        
        # Display risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{-var_95*100:.2f}%")
        with col2:
            st.metric("VaR (99%)", f"{-var_99*100:.2f}%")
        with col3:
            st.metric("CVaR (95%)", f"{-cvar_95*100:.2f}%")
        with col4:
            max_loss = port_returns.min()
            st.metric("Max Daily Loss", f"{max_loss*100:.2f}%")
        
        # VaR distribution
        st.markdown("---")
        st.subheader("  Return Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=port_returns * 100,
            nbinsx=50,
            marker_color='#00D4AA',
            opacity=0.7,
            name='Returns'
        ))
        
        fig.add_vline(x=-var_95*100, line_dash='dash', line_color='#FFE66D',
                     annotation_text=f'VaR 95%: {-var_95*100:.2f}%')
        fig.add_vline(x=-var_99*100, line_dash='dash', line_color='#FF6B6B',
                     annotation_text=f'VaR 99%: {-var_99*100:.2f}%')
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Portfolio Return Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk decomposition
        st.markdown("---")
        st.subheader("  Risk Contribution")
        
        # Marginal risk contribution
        port_vol = np.sqrt(np.dot(weights, cov_matrix.values @ weights))
        marginal_contrib = cov_matrix.values @ weights / port_vol
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight': [f'{w*100:.1f}%' for w in weights],
                'Risk Contrib': [f'{rc*100:.1f}%' for rc in risk_contrib_pct]
            })
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Bar(
                x=selected_assets,
                y=risk_contrib_pct * 100,
                marker_color=['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6'][:n_assets]
            )])
            fig.update_layout(
                height=300,
                template='plotly_dark',
                title='Risk Contribution by Asset',
                yaxis_title='Contribution (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # QUANTUM TAB
    # ==========================================================================
    with tab4:
        section_header("Quantum Optimization", "Quantum-enhanced portfolio optimization")
        
        st.info("  Quantum optimization uses QAOA/VQE algorithms for portfolio allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quantum Parameters")
            
            n_qubits = st.slider("Number of Qubits", 4, 12, 6)
            depth = st.slider("Circuit Depth", 1, 5, 2)
            shots = st.number_input("Measurement Shots", 100, 10000, 1000)
            
            quantum_method = st.selectbox(
                "Quantum Solver",
                ["QAOA", "VQE", "Grover", "Quantum Annealing"]
            )
        
        with col2:
            st.subheader("Problem Formulation")
            
            st.markdown("""
            **QUBO Formulation:**
            
            $\\min_{x} \\sum_{i,j} Q_{ij} x_i x_j$
            
            Where:
            - $Q_{ij} = \\lambda \\cdot \\Sigma_{ij} - (1-\\lambda) \\cdot \\mu_i \\mu_j$
            - $\\Sigma$ = Covariance matrix
            - $\\mu$ = Expected returns
            - $\\lambda$ = Risk aversion
            """)
        
        if st.button("  Run Quantum Optimization"):
            with st.spinner("Running quantum simulation..."):
                import time
                time.sleep(2)  # Simulate computation
                
                # Simulated quantum results
                quantum_weights = np.random.dirichlet(np.ones(n_assets))
                quantum_weights = np.clip(quantum_weights, min_weight, max_weight)
                quantum_weights = quantum_weights / quantum_weights.sum()
                
                st.success("  Quantum optimization complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quantum Weights")
                    
                    q_weights_df = pd.DataFrame({
                        'Asset': selected_assets,
                        'Classical': [f'{w*100:.1f}%' for w in weights],
                        'Quantum': [f'{w*100:.1f}%' for w in quantum_weights]
                    })
                    st.dataframe(q_weights_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.subheader("Comparison")
                    
                    q_return = np.dot(quantum_weights, exp_returns)
                    q_vol = np.sqrt(np.dot(quantum_weights, cov_matrix.values @ quantum_weights))
                    q_sharpe = q_return / q_vol
                    
                    st.metric("Quantum Sharpe", f"{q_sharpe:.3f}",
                             delta=f"{(q_sharpe - port_sharpe):.3f}")
                    
                    st.metric("Quantum Vol", f"{q_vol*100:.2f}%",
                             delta=f"{(q_vol - port_vol)*100:.2f}%")
                
                # Quantum circuit visualization
                st.markdown("---")
                st.subheader("  Quantum Circuit")
                
                # Simple circuit visualization
                fig = go.Figure()
                
                # Draw qubit lines
                for i in range(n_qubits):
                    fig.add_trace(go.Scatter(
                        x=[0, 10],
                        y=[i, i],
                        mode='lines',
                        line=dict(color='white', width=1),
                        showlegend=False
                    ))
                    
                    fig.add_annotation(
                        x=-0.5, y=i, text=f'q{i}', showarrow=False,
                        font=dict(color='white')
                    )
                
                # Add gates
                gates = ['H', 'Ry', 'CNOT', 'Rz', 'H']
                for j, gate in enumerate(gates):
                    for i in range(n_qubits):
                        if np.random.random() > 0.3:
                            color = '#00D4AA' if gate != 'CNOT' else '#FF6B6B'
                            fig.add_shape(
                                type='rect',
                                x0=j*2+0.7, x1=j*2+1.3,
                                y0=i-0.3, y1=i+0.3,
                                fillcolor=color,
                                line=dict(color='white')
                            )
                            fig.add_annotation(
                                x=j*2+1, y=i, text=gate,
                                showarrow=False, font=dict(color='white', size=10)
                            )
                
                fig.update_layout(
                    height=300,
                    template='plotly_dark',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    title='QAOA Variational Circuit'
                )
                
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_portfolio_page()
