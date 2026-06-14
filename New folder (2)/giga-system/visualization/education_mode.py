"""
Education Mode - Interactive Financial Tutorials
===============================================

Interactive educational platform for learning financial concepts,
mathematics, and quantitative finance through guided tutorials
and hands-on exercises.

Features:
- Interactive Black-Scholes tutorial
- Greeks exploration with real-time visualization
- Portfolio optimization learning module
- Monte Carlo simulation concepts
- Options strategies education
- Risk management principles
- Quantum finance introduction
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from core.black_scholes import BlackScholesCalculator
    from core.greeks import GreeksCalculator
    from utils.performance_profiler import PerformanceProfiler
except ImportError:
    # Fallback implementations
    class BlackScholesCalculator:
        def call_price(self, S, K, T, r, sigma):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    class GreeksCalculator:
        def __init__(self):
            self.bs = BlackScholesCalculator()
        
        def delta(self, S, K, T, r, sigma, option_type='call'):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    
    class PerformanceProfiler:
        @staticmethod
        def profile_function(func, *args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start


class EducationMode:
    """Interactive educational platform for financial concepts."""
    
    def __init__(self):
        """Initialize Education Mode."""
        self.bs_calc = BlackScholesCalculator()
        self.greeks_calc = GreeksCalculator()
        self.profiler = PerformanceProfiler()
        
        # Tutorial progress tracking
        if 'tutorial_progress' not in st.session_state:
            st.session_state.tutorial_progress = {}
    
    def mark_tutorial_complete(self, tutorial_name: str):
        """Mark a tutorial as completed."""
        st.session_state.tutorial_progress[tutorial_name] = True
        st.success(f"  {tutorial_name} tutorial completed!")
    
    def get_progress_indicator(self, tutorial_name: str) -> str:
        """Get progress indicator for tutorial."""
        return " " if st.session_state.tutorial_progress.get(tutorial_name, False) else " "
    
    def create_interactive_plot(self, x_data: np.ndarray, y_data: np.ndarray,
                               title: str, x_label: str, y_label: str,
                               highlight_point: Tuple[float, float] = None) -> go.Figure:
        """Create interactive educational plot with highlighting."""
        
        fig = go.Figure()
        
        # Main curve
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=title,
            line=dict(color='blue', width=3)
        ))
        
        # Highlight specific point if provided
        if highlight_point:
            fig.add_trace(go.Scatter(
                x=[highlight_point[0]],
                y=[highlight_point[1]],
                mode='markers',
                name='Current Point',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='circle',
                    line=dict(width=2, color='darkred')
                )
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def black_scholes_tutorial(self):
        """Interactive Black-Scholes tutorial."""
        
        st.subheader("  Black-Scholes Model Tutorial")
        
        # Tutorial introduction
        with st.expander("  What is the Black-Scholes Model?", expanded=True):
            st.markdown("""
            The **Black-Scholes Model** is a mathematical model for pricing options contracts. 
            It was developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s.
            
            **Key Assumptions:**
            - Stock prices follow a geometric Brownian motion
            - Constant risk-free interest rate
            - No dividends during option life
            - European exercise (only at expiration)
            - Constant volatility
            - No transaction costs
            
            **The Formula:**
            
            For a **Call Option:**
            """)
            
            st.latex(r"""
            C = S_0 N(d_1) - K e^{-rT} N(d_2)
            """)
            
            st.markdown("""
            Where:
            - $C$ = Call option price
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $r$ = Risk-free rate
            - $T$ = Time to expiration
            - $N(x)$ = Cumulative standard normal distribution
            """)
            
            st.latex(r"""
            d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}
            """)
            
            st.latex(r"""
            d_2 = d_1 - \sigma\sqrt{T}
            """)
        
        # Interactive parameters
        st.markdown("###  ️ Interactive Parameters")
        st.markdown("Adjust the parameters below to see how they affect the option price:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S = st.slider("Stock Price ($)", 50, 150, 100, 1, 
                         help="Current price of the underlying stock")
            K = st.slider("Strike Price ($)", 50, 150, 100, 1,
                         help="Price at which option can be exercised")
            T = st.slider("Time to Expiry (Years)", 0.01, 2.0, 0.25, 0.01,
                         help="Time remaining until option expires")
        
        with col2:
            r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0, 0.1,
                         help="Risk-free interest rate") / 100
            sigma = st.slider("Volatility (%)", 5.0, 50.0, 20.0, 1.0,
                             help="Expected volatility of the stock") / 100
            option_type = st.selectbox("Option Type", ['call', 'put'])
        
        # Calculate option price
        if option_type == 'call':
            option_price = self.bs_calc.call_price(S, K, T, r, sigma)
        else:
            # Put-call parity for put option
            call_price = self.bs_calc.call_price(S, K, T, r, sigma)
            option_price = call_price - S + K * np.exp(-r * T)
        
        # Display result
        st.markdown("###   Option Price Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Option Price", f"${option_price:.4f}")
        
        with col2:
            intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
            st.metric("Intrinsic Value", f"${intrinsic:.4f}")
        
        with col3:
            time_value = option_price - intrinsic
            st.metric("Time Value", f"${time_value:.4f}")
        
        # Interactive visualization
        st.markdown("###   Price Sensitivity Analysis")
        
        # Stock price sensitivity
        stock_prices = np.linspace(S * 0.7, S * 1.3, 50)
        option_prices = []
        
        for stock_price in stock_prices:
            if option_type == 'call':
                price = self.bs_calc.call_price(stock_price, K, T, r, sigma)
            else:
                call_price = self.bs_calc.call_price(stock_price, K, T, r, sigma)
                price = call_price - stock_price + K * np.exp(-r * T)
            option_prices.append(price)
        
        fig = self.create_interactive_plot(
            stock_prices, option_prices,
            f"{option_type.title()} Option Price vs Stock Price",
            "Stock Price ($)", "Option Price ($)",
            highlight_point=(S, option_price)
        )
        
        # Add payoff diagram
        payoff = []
        for stock_price in stock_prices:
            if option_type == 'call':
                payoff.append(max(0, stock_price - K))
            else:
                payoff.append(max(0, K - stock_price))
        
        fig.add_trace(go.Scatter(
            x=stock_prices,
            y=payoff,
            mode='lines',
            name='Payoff at Expiry',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quiz section
        st.markdown("###   Quick Quiz")
        
        with st.expander("Test Your Understanding"):
            q1 = st.radio(
                "What happens to a call option price when volatility increases?",
                ["Increases", "Decreases", "Stays the same"],
                help="Think about the effect of uncertainty on option value"
            )
            
            if st.button("Check Answer"):
                if q1 == "Increases":
                    st.success("  Correct! Higher volatility increases option value because it increases the chance of favorable price movements.")
                else:
                    st.error("  Try again. Higher volatility is beneficial for option holders.")
            
        # Mark tutorial as complete
        if st.button("  Complete Black-Scholes Tutorial"):
            self.mark_tutorial_complete("Black-Scholes Model")
    
    def greeks_tutorial(self):
        """Interactive Greeks tutorial."""
        
        st.subheader("  Greeks Tutorial - Risk Sensitivities")
        
        # Introduction
        with st.expander("  What are the Greeks?", expanded=True):
            st.markdown("""
            The **Greeks** are risk sensitivities that measure how an option's price changes 
            with respect to various factors. They are essential tools for:
            
            - **Risk Management**: Understanding portfolio sensitivities
            - **Hedging**: Creating delta-neutral or gamma-neutral positions
            - **Trading**: Making informed decisions about option positions
            
            **The Main Greeks:**
            
            1. **Delta (Δ)**: Price sensitivity to underlying price changes
            2. **Gamma (Γ)**: Rate of change of Delta
            3. **Theta (Θ)**: Time decay (sensitivity to time)
            4. **Vega (ν)**: Volatility sensitivity
            5. **Rho (ρ)**: Interest rate sensitivity
            """)
        
        # Interactive Greeks calculator
        st.markdown("###  ️ Interactive Greeks Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S = st.slider("Stock Price ($)", 80, 120, 100, key="greeks_S")
            K = st.slider("Strike Price ($)", 80, 120, 100, key="greeks_K")
            T = st.slider("Time to Expiry (Days)", 1, 365, 90, key="greeks_T") / 365
        
        with col2:
            r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0, key="greeks_r") / 100
            sigma = st.slider("Volatility (%)", 10.0, 50.0, 20.0, key="greeks_sigma") / 100
            option_type = st.selectbox("Option Type", ['call', 'put'], key="greeks_type")
        
        # Calculate Greeks
        delta = self.greeks_calc.delta(S, K, T, r, sigma, option_type)
        
        # Display Greeks
        st.markdown("###   Current Greeks Values")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Delta (Δ)", f"{delta:.4f}",
                     help="Change in option price for $1 change in stock price")
        
        with col2:
            # Simplified gamma calculation
            epsilon = 0.01
            delta_up = self.greeks_calc.delta(S + epsilon, K, T, r, sigma, option_type)
            delta_down = self.greeks_calc.delta(S - epsilon, K, T, r, sigma, option_type)
            gamma = (delta_up - delta_down) / (2 * epsilon)
            st.metric("Gamma (Γ)", f"{gamma:.6f}",
                     help="Change in Delta for $1 change in stock price")
        
        with col3:
            # Simplified theta calculation
            if T > 1/365:  # Avoid division by zero
                option_price_now = self.bs_calc.call_price(S, K, T, r, sigma)
                option_price_tomorrow = self.bs_calc.call_price(S, K, T - 1/365, r, sigma)
                theta = option_price_tomorrow - option_price_now
            else:
                theta = 0
            st.metric("Theta (Θ)", f"{theta:.4f}",
                     help="Change in option price per day (time decay)")
        
        with col4:
            # Simplified vega calculation
            epsilon_vol = 0.01
            price_vol_up = self.bs_calc.call_price(S, K, T, r, sigma + epsilon_vol)
            price_vol_down = self.bs_calc.call_price(S, K, T, r, sigma - epsilon_vol)
            vega = (price_vol_up - price_vol_down) / (2 * epsilon_vol) / 100  # Per 1% vol change
            st.metric("Vega (ν)", f"{vega:.4f}",
                     help="Change in option price per 1% change in volatility")
        
        # Greeks visualization
        st.markdown("###   Greeks Visualization")
        
        # Create subplots for different Greeks
        spot_range = np.linspace(S * 0.8, S * 1.2, 50)
        
        delta_values = [self.greeks_calc.delta(spot, K, T, r, sigma, option_type) 
                       for spot in spot_range]
        
        fig = self.create_interactive_plot(
            spot_range, delta_values,
            f"Delta vs Stock Price ({option_type.title()} Option)",
            "Stock Price ($)", "Delta",
            highlight_point=(S, delta)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Educational insights
        with st.expander("  Greeks Insights"):
            if option_type == 'call':
                st.markdown("""
                **Call Option Greeks Behavior:**
                
                - **Delta**: Ranges from 0 to 1. ATM options have Delta ≈ 0.5
                - **Gamma**: Highest for ATM options, decreases as option moves ITM or OTM
                - **Theta**: Usually negative (time decay). Accelerates as expiration approaches
                - **Vega**: Highest for ATM options with longer time to expiration
                """)
            else:
                st.markdown("""
                **Put Option Greeks Behavior:**
                
                - **Delta**: Ranges from -1 to 0. ATM options have Delta ≈ -0.5
                - **Gamma**: Same as calls - highest for ATM options
                - **Theta**: Usually negative, but can be positive for deep ITM puts
                - **Vega**: Same as calls - highest for ATM options
                """)
        
        # Interactive scenario analysis
        st.markdown("###   Scenario Analysis")
        
        scenario = st.selectbox(
            "Select a scenario to analyze:",
            [
                "Stock price increases by 5%",
                "Volatility increases by 5%", 
                "One week passes",
                "Interest rates increase by 1%"
            ]
        )
        
        if st.button("Analyze Scenario"):
            original_price = self.bs_calc.call_price(S, K, T, r, sigma)
            
            if scenario == "Stock price increases by 5%":
                new_price = self.bs_calc.call_price(S * 1.05, K, T, r, sigma)
                predicted_change = delta * (S * 0.05)
                actual_change = new_price - original_price
                
            elif scenario == "Volatility increases by 5%":
                new_price = self.bs_calc.call_price(S, K, T, r, sigma + 0.05)
                predicted_change = vega * 5  # 5% vol increase
                actual_change = new_price - original_price
                
            elif scenario == "One week passes":
                new_T = max(0.001, T - 7/365)
                new_price = self.bs_calc.call_price(S, K, new_T, r, sigma)
                predicted_change = theta * 7  # 7 days
                actual_change = new_price - original_price
                
            else:  # Interest rate increase
                new_price = self.bs_calc.call_price(S, K, T, r + 0.01, sigma)
                # Simplified rho calculation
                rho_approx = (new_price - original_price) / 0.01
                predicted_change = rho_approx * 0.01
                actual_change = new_price - original_price
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Price", f"${original_price:.4f}")
            with col2:
                st.metric("New Price", f"${new_price:.4f}")
            with col3:
                st.metric("Actual Change", f"${actual_change:.4f}")
            
            if 'predicted_change' in locals():
                accuracy = (1 - abs(actual_change - predicted_change) / abs(actual_change)) * 100
                st.info(f"Greek prediction: ${predicted_change:.4f} (Accuracy: {accuracy:.1f}%)")
        
        if st.button("  Complete Greeks Tutorial"):
            self.mark_tutorial_complete("Greeks")
    
    def portfolio_optimization_tutorial(self):
        """Interactive portfolio optimization tutorial."""
        
        st.subheader("  Portfolio Optimization Tutorial")
        
        with st.expander("  Modern Portfolio Theory", expanded=True):
            st.markdown("""
            **Modern Portfolio Theory (MPT)** was developed by Harry Markowitz in 1952. 
            The key insight is that investors should focus on the risk-return profile of 
            their entire portfolio, not individual securities.
            
            **Key Concepts:**
            
            1. **Diversification**: Spreading investments to reduce risk
            2. **Efficient Frontier**: Set of optimal portfolios for each level of risk
            3. **Risk-Return Tradeoff**: Higher returns generally require higher risk
            4. **Correlation**: How assets move together affects portfolio risk
            
            **Mathematical Foundation:**
            
            Portfolio Return: $R_p = \sum_{i=1}^{n} w_i R_i$
            
            Portfolio Variance: $\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij}$
            
            Where $w_i$ are weights, $R_i$ are returns, and $\sigma_{ij}$ is covariance.
            """)
        
        # Interactive portfolio builder
        st.markdown("###  ️ Build Your Portfolio")
        
        # Asset selection
        num_assets = st.slider("Number of Assets", 2, 5, 3)
        
        # Fetch REAL asset data for educational demonstration
        asset_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'][:num_assets]
        
        try:
            from data.realtime_manager import get_data_manager
            import datetime as dt
            
            dm = get_data_manager()
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=504)  # ~2 years
            
            # Get real historical data for each asset
            expected_returns = []
            volatilities = []
            
            for symbol in asset_names:
                data = dm.get_historical_data_sync(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
                returns = data['close'].pct_change().dropna()
                
                # Annualized statistics
                expected_returns.append(returns.mean() * 252)
                volatilities.append(returns.std() * np.sqrt(252))
            
            expected_returns = np.array(expected_returns)
            volatilities = np.array(volatilities)
        except Exception as e:
            st.error(f"  Real market data unavailable: {e}")
            st.info("  Educational mode requires real market data. Please check your connection.")
            return
        
        # Display asset characteristics
        st.markdown("###   Asset Characteristics")
        
        asset_data = pd.DataFrame({
            'Asset': asset_names,
            'Expected Return': [f"{ret:.2%}" for ret in expected_returns],
            'Volatility': [f"{vol:.2%}" for vol in volatilities],
            'Sharpe Ratio': [f"{(ret - 0.02) / vol:.3f}" for ret, vol in zip(expected_returns, volatilities)]
        })
        
        st.dataframe(asset_data, use_container_width=True, hide_index=True)
        
        # Portfolio weights
        st.markdown("###  ️ Portfolio Weights")
        st.markdown("Adjust the weights (they must sum to 100%):")
        
        weights = []
        cols = st.columns(num_assets)
        
        for i, asset in enumerate(asset_names):
            with cols[i]:
                weight = st.slider(
                    f"{asset} (%)",
                    0, 100, 100 // num_assets,
                    key=f"weight_{i}"
                )
                weights.append(weight / 100)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calculate portfolio metrics
        portfolio_return = np.sum([w * ret for w, ret in zip(weights, expected_returns)])
        
        # Simple correlation matrix (for demonstration)
        correlation_matrix = np.full((num_assets, num_assets), 0.3)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Calculate portfolio volatility
        portfolio_variance = 0
        for i in range(num_assets):
            for j in range(num_assets):
                portfolio_variance += weights[i] * weights[j] * volatilities[i] * volatilities[j] * correlation_matrix[i, j]
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility  # Assume 2% risk-free rate
        
        # Display portfolio metrics
        st.markdown("###   Portfolio Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Return", f"{portfolio_return:.2%}")
        with col2:
            st.metric("Volatility", f"{portfolio_volatility:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio_sharpe:.3f}")
        
        # Portfolio composition pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=asset_names,
            values=[w * 100 for w in weights],
            hole=.3,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig_pie.update_layout(
            title="Portfolio Composition",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Risk-return scatter plot
        st.markdown("###   Risk-Return Analysis")
        
        fig_scatter = go.Figure()
        
        # Plot individual assets
        fig_scatter.add_trace(go.Scatter(
            x=[vol * 100 for vol in volatilities],
            y=[ret * 100 for ret in expected_returns],
            mode='markers+text',
            text=asset_names,
            textposition='top center',
            marker=dict(size=12, color='blue'),
            name='Individual Assets'
        ))
        
        # Plot portfolio
        fig_scatter.add_trace(go.Scatter(
            x=[portfolio_volatility * 100],
            y=[portfolio_return * 100],
            mode='markers+text',
            text=['Portfolio'],
            textposition='top center',
            marker=dict(size=20, color='red', symbol='star'),
            name='Your Portfolio'
        ))
        
        fig_scatter.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Educational insights
        with st.expander("  Diversification Benefits"):
            st.markdown(f"""
            **Your Portfolio Analysis:**
            
            - **Weighted Average Return**: {np.average(expected_returns, weights=weights):.2%}
            - **Portfolio Return**: {portfolio_return:.2%}
            - **Weighted Average Volatility**: {np.average(volatilities, weights=weights):.2%}
            - **Portfolio Volatility**: {portfolio_volatility:.2%}
            
            **Diversification Benefit**: {((np.average(volatilities, weights=weights) - portfolio_volatility) / np.average(volatilities, weights=weights) * 100):.1f}% risk reduction!
            
            This reduction in risk (without reducing expected return) is the key benefit of diversification.
            The portfolio volatility is less than the weighted average of individual asset volatilities 
            due to the correlation structure between assets.
            """)
        
        # Quiz
        with st.expander("  Portfolio Quiz"):
            q1 = st.radio(
                "What happens to portfolio risk as you add more uncorrelated assets?",
                ["Increases", "Decreases", "Stays the same"]
            )
            
            if st.button("Check Portfolio Answer"):
                if q1 == "Decreases":
                    st.success("  Correct! Adding uncorrelated assets reduces portfolio risk through diversification.")
                else:
                    st.error("  Try again. Think about how diversification works.")
        
        if st.button("  Complete Portfolio Optimization Tutorial"):
            self.mark_tutorial_complete("Portfolio Optimization")
    
    def monte_carlo_tutorial(self):
        """Interactive Monte Carlo simulation tutorial."""
        
        st.subheader("  Monte Carlo Simulation Tutorial")
        
        with st.expander("  What is Monte Carlo Simulation?", expanded=True):
            st.markdown("""
            **Monte Carlo Simulation** is a computational technique that uses random sampling 
            to solve mathematical problems and model complex systems.
            
            **In Finance, it's used for:**
            - Option pricing (especially complex derivatives)
            - Risk management (VaR calculations)
            - Portfolio optimization
            - Stress testing
            
            **How it Works:**
            1. Define a mathematical model
            2. Generate random inputs based on probability distributions
            3. Run thousands of simulations
            4. Analyze the results statistically
            
            **Advantages:**
            - Can handle complex, multi-dimensional problems
            - Provides full distribution of outcomes
            - Easy to understand and implement
            """)
        
        # Interactive Monte Carlo demonstration
        st.markdown("###   Interactive Simulation: Stock Price Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            S0 = st.slider("Initial Stock Price ($)", 50, 150, 100)
            mu = st.slider("Expected Return (%/year)", -10, 30, 10) / 100
            sigma = st.slider("Volatility (%/year)", 5, 50, 20) / 100
        
        with col2:
            T = st.slider("Time Horizon (Years)", 0.1, 5.0, 1.0)
            num_sims = st.selectbox("Number of Simulations", [100, 1000, 10000])
            dt = st.slider("Time Steps per Year", 50, 252, 252)
        
        # Run Monte Carlo simulation
        if st.button("  Run Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                
                # Time setup
                num_steps = int(T * dt)
                time_grid = np.linspace(0, T, num_steps + 1)
                
                # Fetch REAL historical data for realistic volatility
                try:
                    from data.realtime_manager import get_data_manager
                    import datetime as dt
                    
                    dm = get_data_manager()
                    end_date = dt.datetime.now()
                    start_date = end_date - dt.timedelta(days=504)
                    
                    spy_data = dm.get_historical_data_sync('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
                    historical_returns = spy_data['close'].pct_change().dropna().values
                    
                    # Bootstrap from real returns
                    paths = np.zeros((num_sims, num_steps + 1))
                    paths[:, 0] = S0
                    
                    for i in range(num_sims):
                        for j in range(1, num_steps + 1):
                            sampled_return = np.random.choice(historical_returns)
                            paths[i, j] = paths[i, j-1] * (1 + sampled_return)
                except Exception as e:
                    st.error(f"  Real historical data unavailable for Monte Carlo: {e}")
                    return
                
                # Create visualization
                fig = go.Figure()
                
                # Plot sample paths (max 50 for readability)
                sample_paths = min(50, num_sims)
                for i in range(sample_paths):
                    fig.add_trace(go.Scatter(
                        x=time_grid,
                        y=paths[i],
                        mode='lines',
                        line=dict(width=1, color='lightblue'),
                        opacity=0.3,
                        showlegend=False
                    ))
                
                # Plot mean path
                mean_path = np.mean(paths, axis=0)
                fig.add_trace(go.Scatter(
                    x=time_grid,
                    y=mean_path,
                    mode='lines',
                    name='Mean Path',
                    line=dict(color='red', width=3)
                ))
                
                # Add confidence intervals
                percentile_95 = np.percentile(paths, 95, axis=0)
                percentile_5 = np.percentile(paths, 5, axis=0)
                
                fig.add_trace(go.Scatter(
                    x=time_grid,
                    y=percentile_95,
                    mode='lines',
                    name='95th Percentile',
                    line=dict(color='green', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=time_grid,
                    y=percentile_5,
                    mode='lines',
                    name='5th Percentile',
                    line=dict(color='orange', dash='dash'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title=f"Monte Carlo Stock Price Simulation ({num_sims:,} paths)",
                    xaxis_title="Time (Years)",
                    yaxis_title="Stock Price ($)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results
                final_prices = paths[:, -1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
                with col2:
                    st.metric("Standard Deviation", f"${np.std(final_prices):.2f}")
                with col3:
                    st.metric("95% VaR", f"${np.percentile(final_prices, 5):.2f}")
                with col4:
                    prob_profit = np.mean(final_prices > S0) * 100
                    st.metric("Probability of Profit", f"{prob_profit:.1f}%")
                
                # Distribution of final prices
                fig_hist = go.Figure(data=[
                    go.Histogram(x=final_prices, nbinsx=50, name='Final Prices')
                ])
                
                fig_hist.update_layout(
                    title="Distribution of Final Stock Prices",
                    xaxis_title="Final Price ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Educational insights
        with st.expander("  Understanding the Results"):
            st.markdown("""
            **Key Insights from Monte Carlo Simulation:**
            
            1. **Mean Reversion**: The average path follows the expected return
            2. **Uncertainty**: Individual paths can deviate significantly from the mean
            3. **Risk Measurement**: We can quantify the probability of different outcomes
            4. **Fat Tails**: Real market distributions often have more extreme outcomes than normal distributions
            
            **Practical Applications:**
            - **Option Pricing**: Especially for path-dependent options
            - **Risk Management**: Calculate Value-at-Risk (VaR) 
            - **Portfolio Planning**: Understand range of possible outcomes
            - **Stress Testing**: Model extreme scenarios
            """)
        
        if st.button("  Complete Monte Carlo Tutorial"):
            self.mark_tutorial_complete("Monte Carlo")
    
    def run_dashboard(self):
        """Run the complete Education Mode dashboard."""
        
        st.title("  Education Mode - Interactive Financial Learning")
        st.markdown("""
        Master quantitative finance concepts through interactive tutorials, 
        real-time visualizations, and hands-on exercises.
        """)
        
        # Progress tracker
        st.sidebar.header("  Learning Progress")
        
        tutorials = [
            "Black-Scholes Model",
            "Greeks",
            "Portfolio Optimization", 
            "Monte Carlo"
        ]
        
        for tutorial in tutorials:
            progress_icon = self.get_progress_indicator(tutorial)
            st.sidebar.markdown(f"{progress_icon} {tutorial}")
        
        completed = len([t for t in tutorials if st.session_state.tutorial_progress.get(t, False)])
        progress_percent = (completed / len(tutorials)) * 100
        st.sidebar.progress(progress_percent / 100)
        st.sidebar.caption(f"Progress: {completed}/{len(tutorials)} tutorials completed ({progress_percent:.0f}%)")
        
        # Main tutorial selection
        st.subheader("  Choose Your Tutorial")
        
        tutorial_tabs = st.tabs([
            "  Black-Scholes", "  Greeks", "  Portfolio Optimization", 
            "  Monte Carlo"
        ])
        
        with tutorial_tabs[0]:
            self.black_scholes_tutorial()
        
        with tutorial_tabs[1]:
            self.greeks_tutorial()
        
        with tutorial_tabs[2]:
            self.portfolio_optimization_tutorial()
        
        with tutorial_tabs[3]:
            self.monte_carlo_tutorial()
        
        # Additional resources
        st.divider()
        
        st.subheader("  Additional Resources")
        
        with st.expander("  Recommended Reading"):
            st.markdown("""
            **Books:**
            - "Options, Futures, and Other Derivatives" by John Hull
            - "Quantitative Risk Management" by McNeil, Frey, and Embrechts
            - "The Concepts and Practice of Mathematical Finance" by Mark Joshi
            
            **Papers:**
            - Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
            - Markowitz (1952): "Portfolio Selection"
            - Merton (1973): "Theory of Rational Option Pricing"
            
            **Online Resources:**
            - QuantLib: Open-source quantitative finance library
            - SSRN: Financial research papers
            - CFA Institute: Professional development
            """)
        
        # Quiz summary
        if completed == len(tutorials):
            st.balloons()
            st.success("  Congratulations! You've completed all tutorials!")
            
            if st.button("  Get Certificate"):
                st.markdown("""
                ---
                ##   Certificate of Completion
                
                **This certifies that you have successfully completed**
                
                # GIGA System Financial Education Program
                
                **Including:**
                - Black-Scholes Model and Option Pricing
                - Greeks and Risk Sensitivities
                - Modern Portfolio Theory and Optimization
                - Monte Carlo Simulation Methods
                
                **Date:** {date}
                
                **Signature:** GIGA Education System
                
                ---
                """.format(date=pd.Timestamp.now().strftime('%B %d, %Y')))


def main():
    """Main function to run the Education Mode."""
    
    st.set_page_config(
        page_title="Education Mode - GIGA System",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for educational styling
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
    .education-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .quiz-correct {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 0.5rem;
    }
    .quiz-incorrect {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run education mode
    education = EducationMode()
    education.run_dashboard()


if __name__ == "__main__":
    main()