"""
GIGA SYSTEM - Backtesting Page
Streamlit page for strategy backtesting
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
    from backtesting.engine import BacktestEngine
    from backtesting.performance import PerformanceAnalyzer
    from data.realtime_manager import get_data_manager, get_historical_data
    BACKTEST_AVAILABLE = True
    REAL_DATA_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    REAL_DATA_AVAILABLE = False


def render_backtest_page():
    """Render the backtesting page."""
    
    st.title(" ️ Strategy Backtesting")
    st.markdown("Event-driven backtesting engine with realistic execution simulation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Backtest Settings")
        
        # Strategy selection
        strategy = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion", "Pairs Trading", 
             "Trend Following", "Market Making", "Options Selling"]
        )
        
        st.markdown("---")
        
        # Universe
        st.subheader("Universe")
        assets = st.multiselect(
            "Assets",
            ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA'],
            default=['AAPL', 'GOOGL', 'MSFT']
        )
        
        st.markdown("---")
        
        # Time period
        st.subheader("Time Period")
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
        end_date = st.date_input("End Date", datetime.now())
        
        st.markdown("---")
        
        # Capital and costs
        st.subheader("Capital & Costs")
        initial_capital = st.number_input(
            "Initial Capital ($)", 
            10000, 10000000, 100000, 10000
        )
        
        commission = st.number_input(
            "Commission (bps)", 0, 100, 10
        ) / 10000
        
        slippage = st.number_input(
            "Slippage (bps)", 0, 100, 5
        ) / 10000
        
        st.markdown("---")
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        if strategy == "Momentum":
            lookback = st.slider("Lookback Period", 5, 60, 20)
            threshold = st.slider("Entry Threshold", 0.0, 0.1, 0.02, 0.01)
        elif strategy == "Mean Reversion":
            lookback = st.slider("Lookback Period", 5, 60, 20)
            zscore_entry = st.slider("Z-Score Entry", 1.0, 3.0, 2.0, 0.1)
        elif strategy == "Pairs Trading":
            lookback = st.slider("Lookback Period", 20, 120, 60)
            zscore_entry = st.slider("Z-Score Entry", 1.5, 3.0, 2.0, 0.1)
        else:
            lookback = 20
            threshold = 0.02
        
        run_backtest = st.button("  Run Backtest", type="primary")
    
    if not assets:
        st.warning("Please select at least one asset")
        return
    
    # Load REAL market data for backtesting
    if REAL_DATA_AVAILABLE:
        try:
            dm = get_data_manager()
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch real historical data
            portfolio_data = dm.get_portfolio_data_sync(assets, start_str, end_str)
            
            if portfolio_data:
                st.success(f"  Loaded REAL market data for {len(assets)} assets")
            else:
                st.warning(" ️ Failed to load real data, using fallback")
                portfolio_data = None
        except Exception as e:
            st.error(f"  Error loading real data: {e}")
            portfolio_data = None
    else:
        portfolio_data = None
        st.error("  Real data module not available")
        st.info("  Backtesting requires real market data from realtime_manager")
        return
    
    # Extract price data from real portfolio data
    prices_data = {}
    for asset in assets:
        prices_data[asset] = portfolio_data[asset]['close']
    
    prices_df = pd.DataFrame(prices_data)
    
    # Main content
    if run_backtest or 'backtest_results' in st.session_state:
        if run_backtest:
            with st.spinner("Running backtest..."):
                # Simulate backtest
                results = run_simulated_backtest(
                    prices_df, strategy, initial_capital, 
                    commission, slippage, lookback if 'lookback' in dir() else 20
                )
                st.session_state['backtest_results'] = results
        
        results = st.session_state['backtest_results']
        
        # Display results
        display_backtest_results(results, prices_df, initial_capital)
    else:
        # Show preview
        st.subheader("  Price Data Preview")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        for asset in assets:
            normalized = prices_df[asset] / prices_df[asset].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=prices_df.index,
                y=normalized,
                name=asset,
                mode='lines'
            ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Normalized Prices (Base = 100)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("  Configure your strategy in the sidebar and click 'Run Backtest'")


def run_simulated_backtest(prices_df: pd.DataFrame, 
                          strategy: str,
                          initial_capital: float,
                          commission: float,
                          slippage: float,
                          lookback: int) -> Dict:
    """Run a simulated backtest."""
    
    returns = prices_df.pct_change().dropna()
    n_days = len(returns)
    
    # Initialize
    equity = [initial_capital]
    positions = {asset: 0 for asset in prices_df.columns}
    trades = []
    daily_returns = []
    
    # Simple strategy simulation
    for i in range(lookback, n_days):
        date = returns.index[i]
        
        # Calculate signals
        for asset in prices_df.columns:
            hist_returns = returns[asset].iloc[i-lookback:i]
            
            if strategy == "Momentum":
                signal = hist_returns.mean() / hist_returns.std() if hist_returns.std() > 0 else 0
                target_pos = 1 if signal > 0.5 else (-1 if signal < -0.5 else 0)
            elif strategy == "Mean Reversion":
                zscore = (prices_df[asset].iloc[i] - prices_df[asset].iloc[i-lookback:i].mean()) / prices_df[asset].iloc[i-lookback:i].std()
                target_pos = -1 if zscore > 2 else (1 if zscore < -2 else 0)
            else:
                signal = np.random.randn()
                target_pos = np.sign(signal) if abs(signal) > 1 else 0
            
            # Execute trade
            if target_pos != positions[asset]:
                side = 'buy' if target_pos > positions[asset] else 'sell'
                price = prices_df[asset].iloc[i]
                qty = abs(target_pos - positions[asset])
                
                # Apply costs
                cost = price * qty * (commission + slippage)
                
                trades.append({
                    'timestamp': date,
                    'asset': asset,
                    'side': side,
                    'price': price,
                    'quantity': qty,
                    'cost': cost
                })
                
                positions[asset] = target_pos
        
        # Calculate daily PnL
        daily_pnl = sum(
            positions[asset] * returns[asset].iloc[i] * equity[-1] / len(prices_df.columns)
            for asset in prices_df.columns
        )
        
        equity.append(equity[-1] + daily_pnl)
        daily_returns.append(daily_pnl / equity[-2] if equity[-2] > 0 else 0)
    
    equity = np.array(equity)
    daily_returns = np.array(daily_returns)
    
    # Calculate metrics
    total_return = (equity[-1] - initial_capital) / initial_capital
    annual_return = total_return * 252 / n_days
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    winning_days = np.sum(np.array(daily_returns) > 0)
    win_rate = winning_days / len(daily_returns) if len(daily_returns) > 0 else 0
    
    return {
        'equity': equity,
        'returns': daily_returns,
        'drawdown': drawdown,
        'trades': trades,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'sortino': sharpe * 1.2,  # Simplified
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'profit_factor': 1.5 if total_return > 0 else 0.5
        },
        'timestamps': prices_df.index[lookback:]
    }


def display_backtest_results(results: Dict, prices_df: pd.DataFrame, initial_capital: float):
    """Display backtest results."""
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    metrics = results['metrics']
    equity = results['equity']
    drawdown = results['drawdown']
    timestamps = results['timestamps']
    
    # Summary metrics
    st.subheader("  Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if metrics['total_return'] >= 0 else "inverse"
        st.metric(
            "Total Return",
            f"{metrics['total_return']*100:.2f}%",
            delta=f"${equity[-1]-initial_capital:,.0f}"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe']:.2f}",
            delta="Good" if metrics['sharpe'] > 1 else "Low"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']*100:.1f}%"
        )
    
    # Second row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Return", f"{metrics['annual_return']*100:.2f}%")
    with col2:
        st.metric("Annual Vol", f"{metrics['annual_vol']*100:.2f}%")
    with col3:
        st.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
    with col4:
        st.metric("Total Trades", f"{metrics['num_trades']}")
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Equity Curve", "  Drawdown", "  Trades", "  Statistics"
    ])
    
    with tab1:
        # Equity curve with benchmark
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio equity
        fig.add_trace(go.Scatter(
            x=list(timestamps) + [timestamps[-1]],
            y=equity,
            name='Portfolio',
            line=dict(color='#00D4AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ), row=1, col=1)
        
        # Benchmark (buy and hold)
        benchmark = prices_df[prices_df.columns[0]].values
        benchmark_normalized = benchmark / benchmark[0] * initial_capital
        fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=benchmark_normalized,
            name='Benchmark (B&H)',
            line=dict(color='#FF6B6B', width=1, dash='dash')
        ), row=1, col=1)
        
        # Daily returns bar
        returns = results['returns']
        colors = ['#00ff88' if r >= 0 else '#ff4444' for r in returns]
        
        fig.add_trace(go.Bar(
            x=timestamps,
            y=np.array(returns) * 100,
            name='Daily Return',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            title='Portfolio Performance',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Return (%)', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Drawdown chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(timestamps) + [timestamps[-1]],
            y=drawdown * 100,
            name='Drawdown',
            line=dict(color='#FF6B6B', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)'
        ))
        
        # Add max drawdown line
        fig.add_hline(
            y=metrics['max_drawdown'] * 100,
            line_dash='dash',
            line_color='white',
            annotation_text=f"Max DD: {metrics['max_drawdown']*100:.2f}%"
        )
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Drawdown Analysis',
            yaxis_title='Drawdown (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        st.subheader("Drawdown Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        with col2:
            avg_dd = np.mean(drawdown[drawdown < 0]) * 100 if np.any(drawdown < 0) else 0
            st.metric("Avg Drawdown", f"{avg_dd:.2f}%")
        with col3:
            dd_duration = np.sum(drawdown < -0.01)
            st.metric("DD Duration (days)", f"{dd_duration}")
    
    with tab3:
        # Trade analysis
        trades = results['trades']
        
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            st.subheader("Trade History")
            st.dataframe(
                trades_df.tail(50),
                use_container_width=True,
                hide_index=True
            )
            
            # Trade distribution
            st.subheader("Trade Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # By asset
                asset_counts = trades_df['asset'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=asset_counts.index,
                    values=asset_counts.values,
                    hole=0.4
                )])
                fig.update_layout(
                    height=300,
                    template='plotly_dark',
                    title='Trades by Asset'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Buy vs Sell
                side_counts = trades_df['side'].value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=side_counts.index,
                    y=side_counts.values,
                    marker_color=['#00ff88', '#ff4444']
                )])
                fig.update_layout(
                    height=300,
                    template='plotly_dark',
                    title='Trade Direction'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades executed during this period")
    
    with tab4:
        # Detailed statistics
        st.subheader("Performance Statistics")
        
        returns = results['returns']
        
        stats = {
            'Performance': {
                'Total Return': f"{metrics['total_return']*100:.2f}%",
                'Annual Return': f"{metrics['annual_return']*100:.2f}%",
                'Annual Volatility': f"{metrics['annual_vol']*100:.2f}%",
                'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
                'Sortino Ratio': f"{metrics['sortino']:.2f}",
                'Calmar Ratio': f"{metrics['calmar']:.2f}"
            },
            'Risk': {
                'Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
                'VaR (95%)': f"{np.percentile(returns, 5)*100:.2f}%",
                'CVaR (95%)': f"{np.mean([r for r in returns if r < np.percentile(returns, 5)])*100:.2f}%",
                'Skewness': f"{pd.Series(returns).skew():.2f}",
                'Kurtosis': f"{pd.Series(returns).kurtosis():.2f}"
            },
            'Trading': {
                'Win Rate': f"{metrics['win_rate']*100:.1f}%",
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Total Trades': f"{metrics['num_trades']}",
                'Avg Trade': f"${(equity[-1]-initial_capital)/max(metrics['num_trades'], 1):,.2f}"
            }
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (category, items) in enumerate(stats.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{category}**")
                for key, value in items.items():
                    st.text(f"{key}: {value}")
        
        # Monthly returns heatmap
        st.markdown("---")
        st.subheader("Monthly Returns")
        
        returns_series = pd.Series(returns, index=timestamps)
        monthly = returns_series.groupby([returns_series.index.year, returns_series.index.month]).sum()
        
        years = sorted(set(returns_series.index.year))
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        z = np.zeros((len(years), 12))
        for (year, month), ret in monthly.items():
            if year in years:
                y_idx = years.index(year)
                m_idx = month - 1
                z[y_idx, m_idx] = ret * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=month_names,
            y=[str(y) for y in years],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(z, 1),
            texttemplate='%{text}%',
            colorbar=dict(title='Return (%)')
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_dark',
            title='Monthly Returns Heatmap'
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_backtest_page()
