"""
GIGA SYSTEM - Options Analytics Page
Streamlit page for options analysis
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
    from core.black_scholes import black_scholes_price, BlackScholesModel
    from core.greeks import OptionGreeks
    from core.implied_volatility import ImpliedVolatility
    from data.realtime_manager import get_data_manager, get_realtime_price
    CORE_AVAILABLE = True
    REAL_DATA_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    REAL_DATA_AVAILABLE = False

try:
    from visualization.charts import (
        volatility_surface, volatility_smile, greeks_chart, payoff_diagram
    )
    from visualization.components import (
        metric_row, section_header, status_indicator
    )
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False


def render_options_page():
    """Render the options analytics page."""
    
    st.title("  Options Analytics")
    st.markdown("Advanced options pricing and Greeks analysis powered by GIGA mathematics")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Option Parameters")
        
        # Add real-time price fetching
        use_realtime = st.checkbox("  Use Real-Time Price", value=False)
        
        if use_realtime and REAL_DATA_AVAILABLE:
            symbol = st.text_input("Underlying Symbol", "AAPL")
            if st.button("  Fetch Real Price"):
                try:
                    dm = get_data_manager()
                    # Try real-time price first
                    real_price = dm.get_realtime_price(symbol)
                    if real_price is None:
                        # Fallback to last close price
                        df = dm.get_historical_data_sync(symbol, 
                            (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                            datetime.now().strftime('%Y-%m-%d')
                        )
                        if not df.empty:
                            real_price = df['close'].iloc[-1]
                    
                    if real_price:
                        st.session_state['spot_price'] = float(real_price)
                        st.success(f"  {symbol} price: ${real_price:.2f}")
                    else:
                        st.error(f"  No data for {symbol}")
                except Exception as e:
                    st.error(f"  Error: {e}")
        
        spot = st.number_input("Spot Price (S)", 50.0, 500.0, 
                              st.session_state.get('spot_price', 100.0), 1.0)
        strike = st.number_input("Strike Price (K)", 50.0, 500.0, 100.0, 1.0)
        time_to_expiry = st.slider("Time to Expiry (days)", 1, 365, 30) / 365
        risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        volatility = st.slider("Volatility (%)", 5, 100, 25) / 100
        dividend = st.number_input("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.1) / 100
        
        option_type = st.radio("Option Type", ["call", "put"])
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Pricing", "  Greeks", "  Volatility Surface", "  Payoff Analysis"
    ])
    
    # ==========================================================================
    # PRICING TAB
    # ==========================================================================
    with tab1:
        section_header("Option Pricing", "Black-Scholes analytical solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("  Call Option")
            
            if CORE_AVAILABLE:
                call_price = black_scholes_price(
                    spot, strike, time_to_expiry, risk_free, volatility, 'call'
                )
            else:
                # Simplified calculation
                from scipy.stats import norm
                d1 = (np.log(spot/strike) + (risk_free + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
                d2 = d1 - volatility*np.sqrt(time_to_expiry)
                call_price = spot*norm.cdf(d1) - strike*np.exp(-risk_free*time_to_expiry)*norm.cdf(d2)
            
            st.metric("Price", f"${call_price:.4f}")
            
            # Intrinsic value
            intrinsic_call = max(spot - strike, 0)
            time_value_call = call_price - intrinsic_call
            
            st.metric("Intrinsic Value", f"${intrinsic_call:.4f}")
            st.metric("Time Value", f"${time_value_call:.4f}")
        
        with col2:
            st.subheader("  Put Option")
            
            if CORE_AVAILABLE:
                put_price = black_scholes_price(
                    spot, strike, time_to_expiry, risk_free, volatility, 'put'
                )
            else:
                from scipy.stats import norm
                d1 = (np.log(spot/strike) + (risk_free + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
                d2 = d1 - volatility*np.sqrt(time_to_expiry)
                put_price = strike*np.exp(-risk_free*time_to_expiry)*norm.cdf(-d2) - spot*norm.cdf(-d1)
            
            st.metric("Price", f"${put_price:.4f}")
            
            # Intrinsic value
            intrinsic_put = max(strike - spot, 0)
            time_value_put = put_price - intrinsic_put
            
            st.metric("Intrinsic Value", f"${intrinsic_put:.4f}")
            st.metric("Time Value", f"${time_value_put:.4f}")
        
        # Put-Call Parity Check
        st.markdown("---")
        st.subheader(" ️ Put-Call Parity Verification")
        
        lhs = call_price + strike * np.exp(-risk_free * time_to_expiry)
        rhs = put_price + spot
        parity_error = abs(lhs - rhs)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("C + K·e^(-rT)", f"${lhs:.4f}")
        with col2:
            st.metric("P + S", f"${rhs:.4f}")
        with col3:
            st.metric("Parity Error", f"${parity_error:.6f}")
            if parity_error < 0.001:
                st.success("  Parity holds!")
    
    # ==========================================================================
    # GREEKS TAB
    # ==========================================================================
    with tab2:
        section_header("Option Greeks", "Sensitivity analysis")
        
        # Calculate Greeks
        if CORE_AVAILABLE:
            greeks_calc = OptionGreeks()
            delta = greeks_calc.delta(spot, strike, time_to_expiry, risk_free, volatility, option_type)
            gamma = greeks_calc.gamma(spot, strike, time_to_expiry, risk_free, volatility)
            theta = greeks_calc.theta(spot, strike, time_to_expiry, risk_free, volatility, option_type)
            vega = greeks_calc.vega(spot, strike, time_to_expiry, risk_free, volatility)
            rho = greeks_calc.rho(spot, strike, time_to_expiry, risk_free, volatility, option_type)
        else:
            from scipy.stats import norm
            d1 = (np.log(spot/strike) + (risk_free + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
            d2 = d1 - volatility*np.sqrt(time_to_expiry)
            
            if option_type == 'call':
                delta = norm.cdf(d1)
                theta = -(spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry))) - risk_free * strike * np.exp(-risk_free * time_to_expiry) * norm.cdf(d2)
                rho = strike * time_to_expiry * np.exp(-risk_free * time_to_expiry) * norm.cdf(d2) / 100
            else:
                delta = norm.cdf(d1) - 1
                theta = -(spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry))) + risk_free * strike * np.exp(-risk_free * time_to_expiry) * norm.cdf(-d2)
                rho = -strike * time_to_expiry * np.exp(-risk_free * time_to_expiry) * norm.cdf(-d2) / 100
            
            gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
            vega = spot * np.sqrt(time_to_expiry) * norm.pdf(d1) / 100
        
        # Display Greeks
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta (Δ)", f"{delta:.4f}")
            st.caption("Price sensitivity")
        
        with col2:
            st.metric("Gamma (Γ)", f"{gamma:.4f}")
            st.caption("Delta sensitivity")
        
        with col3:
            st.metric("Theta (Θ)", f"{theta:.4f}")
            st.caption("Time decay (per day)")
        
        with col4:
            st.metric("Vega (V)", f"{vega:.4f}")
            st.caption("Vol sensitivity")
        
        with col5:
            st.metric("Rho (ρ)", f"{rho:.4f}")
            st.caption("Rate sensitivity")
        
        # Greeks chart across strikes
        st.markdown("---")
        st.subheader("  Greeks Across Strikes")
        
        strikes_range = np.linspace(strike * 0.7, strike * 1.3, 50)
        
        if CORE_AVAILABLE:
            greeks_data = {
                'delta': np.array([greeks_calc.delta(spot, k, time_to_expiry, risk_free, volatility, option_type) for k in strikes_range]),
                'gamma': np.array([greeks_calc.gamma(spot, k, time_to_expiry, risk_free, volatility) for k in strikes_range]),
                'theta': np.array([greeks_calc.theta(spot, k, time_to_expiry, risk_free, volatility, option_type) for k in strikes_range]),
                'vega': np.array([greeks_calc.vega(spot, k, time_to_expiry, risk_free, volatility) for k in strikes_range])
            }
        else:
            # Simplified
            greeks_data = {
                'delta': np.linspace(0.9, 0.1, 50) if option_type == 'call' else np.linspace(-0.1, -0.9, 50),
                'gamma': np.exp(-((strikes_range - spot)**2) / (2 * (spot * 0.2)**2)),
                'theta': np.linspace(-0.05, -0.02, 50),
                'vega': np.exp(-((strikes_range - spot)**2) / (2 * (spot * 0.3)**2)) * 0.3
            }
        
        if CHARTS_AVAILABLE:
            fig = greeks_chart(strikes_range, greeks_data, spot)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=['Delta', 'Gamma', 'Theta', 'Vega'])
            
            fig.add_trace(go.Scatter(x=strikes_range, y=greeks_data['delta'], name='Delta'), row=1, col=1)
            fig.add_trace(go.Scatter(x=strikes_range, y=greeks_data['gamma'], name='Gamma'), row=1, col=2)
            fig.add_trace(go.Scatter(x=strikes_range, y=greeks_data['theta'], name='Theta'), row=2, col=1)
            fig.add_trace(go.Scatter(x=strikes_range, y=greeks_data['vega'], name='Vega'), row=2, col=2)
            
            fig.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # VOLATILITY SURFACE TAB
    # ==========================================================================
    with tab3:
        section_header("Volatility Surface", "3D implied volatility visualization")
        
        st.info("  This tab shows the implied volatility surface across strikes and maturities")
        
        # Generate synthetic vol surface
        strikes_pct = np.linspace(0.8, 1.2, 20)  # Moneyness
        maturities = np.array([7, 14, 30, 60, 90, 180, 365])  # Days
        
        # Create vol surface with skew
        base_vol = volatility
        vol_surface = np.zeros((len(maturities), len(strikes_pct)))
        
        for i, mat in enumerate(maturities):
            for j, k_pct in enumerate(strikes_pct):
                # Skew: higher vol for OTM puts, lower for OTM calls
                skew = 0.1 * (1 - k_pct)  # Negative for OTM calls, positive for OTM puts
                term_structure = 0.02 * np.log(mat / 30)  # Term structure
                smile = 0.05 * (k_pct - 1) ** 2  # Smile effect
                
                vol_surface[i, j] = base_vol + skew + term_structure + smile
        
        # 3D Surface
        if CHARTS_AVAILABLE:
            fig = volatility_surface(
                strikes_pct * spot,
                maturities,
                vol_surface
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Surface(
                x=strikes_pct * spot,
                y=maturities,
                z=vol_surface * 100,
                colorscale='Viridis'
            )])
            fig.update_layout(
                height=600,
                template='plotly_dark',
                scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Maturity (days)',
                    zaxis_title='IV (%)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Volatility Smile
        st.markdown("---")
        st.subheader("  Volatility Smile")
        
        selected_maturity = st.select_slider(
            "Select Maturity",
            options=maturities,
            value=30
        )
        
        mat_idx = list(maturities).index(selected_maturity)
        smile_ivs = vol_surface[mat_idx, :]
        
        if CHARTS_AVAILABLE:
            fig = volatility_smile(strikes_pct * spot, smile_ivs, spot)
            st.plotly_chart(fig, use_container_width=True)
        else:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=strikes_pct,
                y=smile_ivs * 100,
                mode='lines+markers',
                name='IV'
            ))
            fig.add_vline(x=1.0, line_dash='dash', annotation_text='ATM')
            fig.update_layout(
                height=400,
                template='plotly_dark',
                xaxis_title='Moneyness (K/S)',
                yaxis_title='IV (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # PAYOFF TAB
    # ==========================================================================
    with tab4:
        section_header("Payoff Analysis", "Strategy payoff diagrams")
        
        strategy = st.selectbox(
            "Select Strategy",
            ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread",
             "Long Straddle", "Iron Condor", "Butterfly"]
        )
        
        # Generate spot price range
        spot_range = np.linspace(spot * 0.5, spot * 1.5, 100)
        
        # Calculate payoffs based on strategy
        if strategy == "Long Call":
            if CORE_AVAILABLE:
                premium = black_scholes_price(spot, strike, time_to_expiry, risk_free, volatility, 'call')
            else:
                premium = call_price
            payoff = np.maximum(spot_range - strike, 0) - premium
            
        elif strategy == "Long Put":
            if CORE_AVAILABLE:
                premium = black_scholes_price(spot, strike, time_to_expiry, risk_free, volatility, 'put')
            else:
                premium = put_price
            payoff = np.maximum(strike - spot_range, 0) - premium
            
        elif strategy == "Bull Call Spread":
            strike_low = strike * 0.95
            strike_high = strike * 1.05
            payoff = np.maximum(spot_range - strike_low, 0) - np.maximum(spot_range - strike_high, 0) - 2
            
        elif strategy == "Bear Put Spread":
            strike_low = strike * 0.95
            strike_high = strike * 1.05
            payoff = np.maximum(strike_high - spot_range, 0) - np.maximum(strike_low - spot_range, 0) - 2
            
        elif strategy == "Long Straddle":
            payoff = np.maximum(spot_range - strike, 0) + np.maximum(strike - spot_range, 0) - call_price - put_price
            
        elif strategy == "Iron Condor":
            k1, k2, k3, k4 = strike * 0.9, strike * 0.95, strike * 1.05, strike * 1.1
            payoff = (np.maximum(spot_range - k1, 0) - np.maximum(spot_range - k2, 0) -
                     np.maximum(spot_range - k3, 0) + np.maximum(spot_range - k4, 0) + 3)
            
        elif strategy == "Butterfly":
            k1, k2, k3 = strike * 0.95, strike, strike * 1.05
            payoff = (np.maximum(spot_range - k1, 0) - 2 * np.maximum(spot_range - k2, 0) +
                     np.maximum(spot_range - k3, 0) - 1)
        else:
            payoff = np.zeros_like(spot_range)
        
        # Plot payoff
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoff,
            mode='lines',
            name='Payoff',
            line=dict(color='#00D4AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
        fig.add_vline(x=spot, line_dash='dot', line_color='yellow',
                     annotation_text='Current Spot')
        
        # Breakeven points
        breakeven = spot_range[np.abs(payoff) < 0.5]
        for be in breakeven[:2]:  # Show max 2 breakeven points
            fig.add_vline(x=be, line_dash='dash', line_color='#FF6B6B', opacity=0.5)
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            title=f'{strategy} Payoff Diagram',
            xaxis_title='Spot Price at Expiry',
            yaxis_title='Profit/Loss ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Profit", f"${np.max(payoff):.2f}")
        with col2:
            st.metric("Max Loss", f"${np.min(payoff):.2f}")
        with col3:
            be_points = spot_range[np.abs(payoff) < 0.5]
            if len(be_points) > 0:
                st.metric("Breakeven", f"${be_points[0]:.2f}")
            else:
                st.metric("Breakeven", "N/A")
        with col4:
            st.metric("Risk/Reward", f"{abs(np.min(payoff)/max(np.max(payoff), 0.01)):.2f}x")


if __name__ == "__main__":
    render_options_page()
