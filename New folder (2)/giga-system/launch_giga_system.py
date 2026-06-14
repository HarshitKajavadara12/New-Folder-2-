#!/usr/bin/env python3
"""
GIGA SYSTEM - LIVE EXECUTION ENGINE (PHASE 13)
The 'Reality' Pipeline. 
THIS SCRIPT IS THE ONLY AUTHORIZED LIVE RUNNER.

Responsibilities:
1. Load the 'strategies_config.toml' Artifact (The Truth).
2. Initialize Component Drivers (The Reality).
3. Execute the Loop blindly based on Artifact parameters.
"""

import sys
import time
import toml
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# --- PROD IMPORTS (NO RESEARCH ORCHESTRATORS) ---
try:
    from data.live.market_stream import MarketStream
    from execution.binance_executor import BinanceExecutor
    from account.live_account import LiveAccount
    from session.session_controller import SessionController
    
    # We import the strategy implementation, but its initialization 
    # MUST be driven by the TOML config, not hardcoded values.
    from research.strategies.momentum import LiveMomentumStrategy 
    from research.core.greek_response import VariationalAnalyzer
except ImportError as e:
    print(f"  LIVE ENVIRONMENT BROKEN: Missing Component: {e}")
    sys.exit(1)

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [LIVE] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/live_engine.log")
    ]
)
logger = logging.getLogger("GIGA_LIVE")

def main():
    print("\n" + "="*60)
    print("  LAUNCHING GIGA SYSTEM: LIVE ENGINE")
    print("   PHASE 13: STRICT SEPARATION - REALITY DATA")
    print("="*60)

    # 1. LOAD CONFIGURATION (THE BRIDGE)
    # ----------------------------------------------------
    config_path = Path("config/strategies_config.toml")
    if not config_path.exists():
        logger.critical("ARTIFACT MISSING: config/strategies_config.toml")
        logger.critical("  STOP: Run 'demo_complete_system.py' (Research) first to generate parameters.")
        sys.exit(1)
        
    try:
        bridge_config = toml.load(config_path)
        meta = bridge_config.get('meta', {})
        logger.info(f"[OK] Bridge Loaded. Source: {meta.get('source', 'Unknown')}")
        logger.info(f"     Generated At: {meta.get('generated_at', 'Unknown')}")
        logger.info(f"     Target Kappa: {meta.get('kappa_score', 'N/A')}")
    except Exception as e:
        logger.critical(f"Bridge Corrupted: {e}")
        sys.exit(1)

    # 2. INITIALIZE LIVE COMPONENTS
    # ----------------------------------------------------
    # A. Account & Risk
    try:
        start_bal = 10000.0
        account = LiveAccount(start_balance=start_bal) 
        session = SessionController()
        session.begin(account.equity)
        logger.info(f"[RISK] Global Risk Controller Active. Equity: ${account.equity:,.2f}")
    except Exception as e:
        logger.critical(f"Risk Init Failed: {e}")
        sys.exit(1)
    
    # B. Execution Stub (Paper Mode Verified)
    exec_conf = bridge_config.get("execution_params", {})
    executor = BinanceExecutor(
        api_key="LIVE_KEY_PLACEHOLDER", 
        paper_mode=True # FORCE PAPER IN PHASE 13
    )
    # Configure Execution based on Bridge
    slippage_limit = exec_conf.get('max_slippage_bps', 10)
    logger.info(f"[EXEC] Executor Online. Slippage Limit: {slippage_limit}bps")

    # C. Strategy Engine
    # Note: In a pure production system, we might inject params here.
    strat = LiveMomentumStrategy()
    logger.info("[STRAT] LIVE MOMENTUM STRATEGY: MOUNTED")
    
    # D. Live History (For Greeks)
    import pandas as pd
    import numpy as np
    history_prices = []
    
    # --- PHASE 14 STATE MACHINE REFINEMENTS ---
    position_state = "FLAT" # FLAT, LONG, SHORT
    last_trade_ts = 0
    min_trade_interval = 2.0 # Seconds (HFT Cooldown)
    last_valid_delta = 0.0   # Greek Guard
    
    print("-" * 60)
    print("  SYSTEM READY. WAITING FOR MARKET DATA...")
    print("-" * 60)

    # 3. REALTIME LOOP (The Hot Path)
    # ----------------------------------------------------
    def on_tick(tick):
        nonlocal position_state, last_trade_ts, last_valid_delta
        
        # FAST PATH START
        t0 = time.perf_counter()
        
        price = tick.get('price')
        ts = tick.get('timestamp')
        
        if not price: return
        
        # Capture History for Greeks
        history_prices.append(price)
        if len(history_prices) > 50: history_prices.pop(0)

        # A. Session Guard (Risk)
        # ------------------------
        if not session.heartbeat(account.equity):
            logger.warning("[STOP] SESSION GUARD: STOP LIMIT HIT. Halted.")
            stream.stop()
            sys.exit(0)

        # B. Bridge Logic / Regime Selection
        # ------------------------
        regime_params = bridge_config.get("regime_params", {}).get("LOW_VOL", {})
        target_leverage = regime_params.get("leverage", 1.0)
        target_kappa = meta.get('kappa_score', 0)
        
        # C. Greek Calculation (Live Guarded)
        # -----------------------
        current_delta = last_valid_delta # Default to last known good
        if len(history_prices) > 10:
            try:
                # Simplified Local Delta (Correlation of change)
                s_series = pd.Series(history_prices)
                pnl_proxy = s_series.diff().cumsum().fillna(0)
                
                raw_delta = VariationalAnalyzer.calculate_delta(pnl_proxy, s_series)
                
                #   FIX 1: Greek Math Guard
                if not np.isnan(raw_delta) and not np.isinf(raw_delta):
                    current_delta = raw_delta
                    last_valid_delta = raw_delta
                # else: keep using last_valid_delta (Logic: "Greek math must never block execution")
            except Exception:
                pass # Swallow math errors to keep engine running
        
        # D. Strategy Update
        # ------------------------
        signal = strat.update(price, ts)
        
        # LOGGING BLOCK
        # ------------------------
        # 1. Greeks
        logger.info(f"[MATH] Price:{price:.1f} | D(Delta):{current_delta:.2f} | K(target):{target_kappa}")
        
        # 2. Strategy
        if signal:
            logger.info(f"[SENSE] {signal.get('action')} | Conf:{signal.get('confidence')} | Reason:{signal.get('reason')}")

        # E. Execution Logic (Guarded)
        # ------------------------
        if signal and signal.get('action') in ['ENTER_LONG', 'ENTER_SHORT']:
            
            # Determine Action Direction
            is_long = "LONG" in signal['action']
            action_side = "BUY" if is_long else "SELL"
            target_state = "LONG" if is_long else "SHORT"
            
            #   FIX 2: Position State Lock
            if position_state == target_state:
                logger.info(f"[SKIP] State Lock: Already {position_state}. Ignoring signal.")
                sys.stdout.write(f"\r[LIVE] {price:,.2f} | Eq: {account.equity:,.0f} | State: {position_state}\n")
                return

            #   FIX 3: Cooldown Window
            now_ts = time.time()
            if (now_ts - last_trade_ts) < min_trade_interval:
                logger.warning(f"[SKIP] Cooldown Active: {now_ts - last_trade_ts:.2f}s < {min_trade_interval}s")
                return

            # 3. Risk Validations
            logger.info(f"[RISK] Validating Signal: {action_side}...")
            
            if target_kappa < 5.0: 
                logger.warning(f"[RISK] REJECT: Kappa ({target_kappa}) too low for Momentum.")
                return 

            size_usd = account.equity * 0.1 * target_leverage
            size_btc = size_usd / price
            
            logger.info(f"[RISK] APPROVED. Exp: {target_leverage}x | Size: {size_btc:.4f} BTC")
            
            # Construct Order
            order = {
                "symbol": "BTCUSDT",
                "side": action_side,
                "quantity": size_btc,
                "type": "MARKET"
            }
            
            # Execute
            logger.info(f"[EXEC] Sending Order -> BinanceExecutor...")
            exec_result = executor.execute_order(order)
            
            #   FIX 4: Update State & Handle Chaos
            if exec_result.get('status') == 'FILLED':
                fill_price = exec_result['avg_price']
                fill_qty = exec_result['executed_qty']
                account.execute_trade("BTCUSDT", action_side, fill_price, fill_qty)
                
                # Update State
                position_state = target_state
                last_trade_ts = now_ts
                
                logger.info(f"[FILL] {action_side} @ {fill_price:.2f} (Latency: {exec_result.get('latency_ms', 0)}ms)")
                logger.info(f"[STATE] Transition: {position_state} -> {target_state}")
            else:
                logger.warning(f"[EXEC] REJECTED by Exchange/Chaos: {exec_result.get('reason', 'Unknown')}")
                # Do NOT update state. Ensure we retry or wait for next clean signal.

        # E. Feedback / UI (Sampled)
        # ------------------------
        # Log to console so user sees scrolling history (requested fix)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Only log every few ticks to prevent clutter if high freq, or just log everything.
        # Given "one line" complaint, we log explicitly with newlines.
        logger.info(f"[TICK] {price:,.2f} | Eq: {account.equity:,.0f} | Lat: {elapsed_ms:.2f}ms")

    # 4. CONNECT FEED & SPIN
    # ----------------------------------------------------
    stream = MarketStream("BTCUSDT", callback=on_tick)
    
    try:
        stream.start()
        # Keep main thread blocking while async stream runs in background thread
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n  MANUAL OVERRIDE: STOPPING SYSTEM")
        stream.stop()
        print("System Halted Safely.")

if __name__ == "__main__":
    main()
