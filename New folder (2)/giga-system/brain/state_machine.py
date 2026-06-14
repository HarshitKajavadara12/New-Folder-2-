"""
GIGA SYSTEM - State Machine Brain
Implementation: Finite State Machine (FSM) for Strategy Logic

REALITY CHECK:
- Implements strict States (IDLE, ANALYZING, ENTRY, MANAGING, EXIT)
- Enforces COOLDOWNS
- Manages REGIME transitions
- No "If logic" -> "State Transitions"
"""

from enum import Enum, auto
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class State(Enum):
    BOOT = auto()
    IDLE = auto()
    ANALYZING = auto()
    ENTRY_SIGNAL = auto()
    IN_POSITION = auto()
    EXIT_SIGNAL = auto()
    COOLDOWN = auto()
    HALTED = auto() # Risk Trip

class MarketRegime(Enum):
    LOW_VOL = auto()
    NORMAL_VOL = auto()
    HIGH_VOL = auto()
    CRASH = auto()

class StateMachineBrain:
    def __init__(self, config: Dict):
        self.state = State.BOOT
        self.regime = MarketRegime.NORMAL_VOL
        self.last_transition = time.time()
        self.cooldown_expires = 0
        self.config = config
        
        # Position Memory
        self.active_position = None  # {side, size, entry, stop}
        
        # Exit Management (BUG#1 FIX: Real exit logic)
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.high_water_mark = 0.0  # For trailing stop
        self.low_water_mark = float('inf')
        self.bars_in_position = 0
        self.max_bars_in_position = config.get('max_holding_bars', 200)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.02)  # 2%
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3%
        self.take_profit_pct = config.get('take_profit_pct', 0.06)  # 6%
        self.cooldown_seconds = config.get('cooldown_seconds', 5.0)
    
    def update(self, market_data: Dict) -> Optional[Dict]:
        """
        Main Brain Tick. returns 'Action' or None.
        """
        now = time.time()
        iv = market_data.get('iv', 0.5)
        price = market_data.get('price', 0.0)
        
        # 1. Regime Detection
        self._detect_regime(iv)
        
        # 2. State Logic
        if self.state == State.BOOT:
            self._transition(State.IDLE)
            
        elif self.state == State.IDLE:
            if self.regime == MarketRegime.CRASH:
                self._transition(State.HALTED)
            else:
                self._transition(State.ANALYZING)
                
        elif self.state == State.ANALYZING:
            signal = self._evaluate_entry(market_data)
            if signal:
                self._transition(State.ENTRY_SIGNAL)
                return {"action": "PREPARE_ORDER", "signal": signal}
                
        elif self.state == State.ENTRY_SIGNAL:
            # Waiting for execution confirmation
            self.state = State.IN_POSITION
            # Record entry for exit logic
            if self.active_position:
                side = self.active_position.get('side', 'LONG')
                self.entry_price = price
                self.high_water_mark = price
                self.low_water_mark = price
                self.bars_in_position = 0
                if 'LONG' in str(side).upper():
                    self.stop_price = price * (1.0 - self.stop_loss_pct)
                    self.target_price = price * (1.0 + self.take_profit_pct)
                else:
                    self.stop_price = price * (1.0 + self.stop_loss_pct)
                    self.target_price = price * (1.0 - self.take_profit_pct)
            return {"action": "EXECUTE_ENTRY", "params": self.active_position}

        elif self.state == State.IN_POSITION:
            exit_sig = self._evaluate_exit(market_data)
            if exit_sig:
                self._transition(State.EXIT_SIGNAL)
                return {"action": "EXECUTE_EXIT", "reason": exit_sig}

        elif self.state == State.EXIT_SIGNAL:
             self._transition(State.COOLDOWN)
             self.cooldown_expires = now + self.cooldown_seconds
             
        elif self.state == State.COOLDOWN:
            if now > self.cooldown_expires:
                self._transition(State.IDLE)
                
        elif self.state == State.HALTED:
            # Manual reset required or extreme safety check
            if iv < 1.0: # Vol drop
                self._transition(State.IDLE)

        return None

    def _transition(self, new_state: State):
        logger.info(f"[BRAIN] State: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.last_transition = time.time()

    def _detect_regime(self, iv: float):
        if iv > 2.0: self.regime = MarketRegime.CRASH
        elif iv > 1.0: self.regime = MarketRegime.HIGH_VOL
        elif iv < 0.3: self.regime = MarketRegime.LOW_VOL
        else: self.regime = MarketRegime.NORMAL_VOL

    def _evaluate_entry(self, tick) -> Optional[str]:
        # PHASE 8: Intelligence Hook
        # If the Intelligence Layer provided a decision, use it.
        ai_decision = tick.get('ai_decision')
        if ai_decision:
            # Action format is "ENTRY_LONG" or "ENTRY_SHORT"
            return ai_decision['action']

        # Fallback (Legacy Logic)
        delta = tick.get('delta', 0.5)
        
        # Strategy logic based on Regime
        if self.regime == MarketRegime.LOW_VOL:
             # In low vol, we want directional exposure with cheap options
             if delta > 0.6: return "LONG_CALL_MOMENTUM"
             
        elif self.regime == MarketRegime.NORMAL_VOL:
             # In normal vol, we trade standard deltas
             if delta > 0.52: return "LONG_CALL_TREND"
             if delta < 0.48: return "SHORT_CALL_TREND"
             
        elif self.regime == MarketRegime.HIGH_VOL:
             # In high vol, we might sell premium or wait
             pass
             
        return None

    def _evaluate_exit(self, tick) -> Optional[str]:
        """
        Evaluate exit conditions: stop-loss, take-profit, trailing stop, time-based.
        Returns exit reason string or None.
        """
        price = tick.get('price', 0.0)
        if price <= 0 or self.entry_price <= 0:
            return None
        
        self.bars_in_position += 1
        
        # Determine position direction
        side = 'LONG'
        if self.active_position:
            pos_action = str(self.active_position.get('side', self.active_position.get('action', 'LONG')))
            if 'SHORT' in pos_action.upper() or 'SELL' in pos_action.upper():
                side = 'SHORT'
        
        if side == 'LONG':
            # Update trailing stop high-water mark
            if price > self.high_water_mark:
                self.high_water_mark = price
                # Ratchet stop up
                trailing_stop = self.high_water_mark * (1.0 - self.trailing_stop_pct)
                if trailing_stop > self.stop_price:
                    self.stop_price = trailing_stop
            
            # 1. Hard Stop-Loss
            if price <= self.stop_price:
                return f"STOP_LOSS at {price:.2f} (stop={self.stop_price:.2f})"
            
            # 2. Take-Profit
            if price >= self.target_price:
                return f"TAKE_PROFIT at {price:.2f} (target={self.target_price:.2f})"
                
        else:  # SHORT
            # Update trailing stop low-water mark
            if price < self.low_water_mark:
                self.low_water_mark = price
                trailing_stop = self.low_water_mark * (1.0 + self.trailing_stop_pct)
                if trailing_stop < self.stop_price:
                    self.stop_price = trailing_stop
            
            # 1. Hard Stop-Loss (price rises above stop for shorts)
            if price >= self.stop_price:
                return f"STOP_LOSS at {price:.2f} (stop={self.stop_price:.2f})"
            
            # 2. Take-Profit (price drops below target for shorts)
            if price <= self.target_price:
                return f"TAKE_PROFIT at {price:.2f} (target={self.target_price:.2f})"
        
        # 3. Time-based exit (max holding period)
        if self.bars_in_position >= self.max_bars_in_position:
            return f"TIME_EXIT after {self.bars_in_position} bars"
        
        # 4. Regime change exit (market crash while in position)
        if self.regime == MarketRegime.CRASH:
            return f"REGIME_EXIT: CRASH detected (IV surge)"
        
        # 5. AI-driven exit signal
        ai_decision = tick.get('ai_decision')
        if ai_decision and 'EXIT' in str(ai_decision.get('action', '')).upper():
            return f"AI_EXIT: {ai_decision.get('reason', 'model signal')}"
        
        return None

    def get_state_summary(self) -> Dict:
        """Get current state summary for monitoring and logging."""
        summary = {
            'state': self.state.name,
            'regime': self.regime.name,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'high_water_mark': self.high_water_mark,
            'low_water_mark': self.low_water_mark,
            'bars_in_position': self.bars_in_position,
            'last_transition': self.last_transition,
            'cooldown_expires': self.cooldown_expires,
            'has_position': self.active_position is not None,
        }
        if self.active_position:
            summary['position_side'] = self.active_position.get('side', 'UNKNOWN')
        return summary

    def reset_position(self):
        """Reset all position-related state (used after exit)."""
        self.active_position = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.high_water_mark = 0.0
        self.low_water_mark = float('inf')
        self.bars_in_position = 0
