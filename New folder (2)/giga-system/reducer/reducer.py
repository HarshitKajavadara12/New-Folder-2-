"""
GIGA SYSTEM - The Decision Reducer (Phase 8)
LAYER 3: AUTHORITY

Aggregates signals from multiple strategies, applies regime filters,
and issues the Final Verdict: ENTRY_LONG, ENTRY_SHORT, EXIT, or HOLD.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionReducer:
    """
    The Brain's Core Processor.
    Aggregates signals, checks context, and issues the Final Verdict.
    
    Supports:
    - Weighted vote aggregation across strategies
    - Regime-adaptive confidence thresholds
    - EXIT signal generation (from position + reversal signals)
    - Conflict resolution (opposing signals cancel out)
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Weighting Configuration
        self.strategy_weights = config.get("strategy_weights", {
            "Momentum_V1": 1.0,
            "MarketMaking_V1": 0.5,
            "PairsTrading_V1": 0.8,
            "VolArb_V1": 0.7,
        })
        self.min_confidence = 0.6
        self.exit_confidence = 0.4  # Lower threshold for exits (favor safety)
        self.last_decision = None
        self.decision_count = 0
        
        # Decision history for audit/analysis
        self.decision_history: List[Dict] = []
        self.max_history: int = config.get('max_history', 100)
        
        # Position tracking for exit logic
        self.current_position: Optional[Dict] = None  # {side: "LONG"/"SHORT", symbol: str}

    def set_position(self, position: Optional[Dict]):
        """Update the reducer's knowledge of current position."""
        self.current_position = position

    def decide(self, signals: List[Dict], market_state: str) -> Optional[Dict]:
        """
        Input: List of signals from strategies + current market state
        Output: Final executable Signal or None
        
        Signal format: {source, action, confidence, ...}
        Actions: "ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT", "HOLD"
        """
        if not signals:
            return None

        # 1. Regime-Adaptive Thresholds
        if market_state == "HIGH_VOL":
            self.min_confidence = 0.8
            self.exit_confidence = 0.3  # Even more eager to exit in high vol
        elif market_state == "CRASH":
            self.min_confidence = 0.95  # Almost never enter in crash
            self.exit_confidence = 0.2  # Very eager to exit
        else:
            self.min_confidence = 0.6
            self.exit_confidence = 0.4

        # 2. Separate entry signals from exit signals
        entry_signals = []
        exit_signals = []
        
        for sig in signals:
            action = sig.get('action', 'HOLD')
            if 'EXIT' in action:
                exit_signals.append(sig)
            elif action != 'HOLD':
                entry_signals.append(sig)

        # 3. Process EXIT signals first (safety priority)
        exit_decision = self._process_exit_signals(exit_signals)
        if exit_decision:
            return exit_decision

        # 4. Check for reversal (in position but signals reversed)
        reversal = self._check_reversal(entry_signals)
        if reversal:
            return reversal

        # 5. Process ENTRY signals
        return self._process_entry_signals(entry_signals)
    
    def _process_exit_signals(self, signals: List[Dict]) -> Optional[Dict]:
        """Process explicit exit signals."""
        if not signals or not self.current_position:
            return None
        
        exit_score = 0.0
        total_weight = 0.0
        reasons = []
        
        for sig in signals:
            source = sig.get('source', 'Unknown')
            weight = self.strategy_weights.get(source, 0.5)
            conf = sig.get('confidence', 0.5)
            
            exit_score += conf * weight
            total_weight += weight
            reasons.append(f"{source}:EXIT")
        
        normalized = exit_score / max(total_weight, 1.0)
        
        if normalized > self.exit_confidence:
            side = self.current_position.get('side', 'LONG')
            exit_action = f"EXIT_{side}"
            self.decision_count += 1
            decision = {
                "action": exit_action,
                "size": 1.0,  # Full exit
                "reason": f"Exit consensus: {reasons} (Score: {normalized:.2f})",
                "decision_id": self.decision_count
            }
            self.last_decision = decision
            self._record_decision(decision)
            logger.info(f"[REDUCER] {exit_action}: {decision['reason']}")
            return decision
        
        return None
    
    def _check_reversal(self, entry_signals: List[Dict]) -> Optional[Dict]:
        """If we're long and signals say short (or vice versa), issue EXIT."""
        if not self.current_position or not entry_signals:
            return None
        
        current_side = self.current_position.get('side', '').upper()
        
        # Count direction votes
        long_votes = 0
        short_votes = 0
        
        for sig in entry_signals:
            action = sig.get('action', 'HOLD')
            conf = sig.get('confidence', 0.5)
            source = sig.get('source', 'Unknown')
            weight = self.strategy_weights.get(source, 0.5)
            
            if 'LONG' in action or 'BUY' in action:
                long_votes += conf * weight
            elif 'SHORT' in action or 'SELL' in action:
                short_votes += conf * weight
        
        # Reversal detection
        if current_side == 'LONG' and short_votes > long_votes and short_votes > self.exit_confidence:
            self.decision_count += 1
            decision = {
                "action": "EXIT_LONG",
                "size": 1.0,
                "reason": f"Reversal detected: short_score={short_votes:.2f} > long_score={long_votes:.2f}",
                "decision_id": self.decision_count
            }
            self.last_decision = decision
            self._record_decision(decision)
            logger.info(f"[REDUCER] REVERSAL EXIT_LONG: {decision['reason']}")
            return decision
        
        if current_side == 'SHORT' and long_votes > short_votes and long_votes > self.exit_confidence:
            self.decision_count += 1
            decision = {
                "action": "EXIT_SHORT",
                "size": 1.0,
                "reason": f"Reversal detected: long_score={long_votes:.2f} > short_score={short_votes:.2f}",
                "decision_id": self.decision_count
            }
            self.last_decision = decision
            self._record_decision(decision)
            logger.info(f"[REDUCER] REVERSAL EXIT_SHORT: {decision['reason']}")
            return decision
        
        return None
    
    def _process_entry_signals(self, signals: List[Dict]) -> Optional[Dict]:
        """Standard vote aggregation for entry signals."""
        vote_score = 0.0
        total_weight = 0.0
        reasons = []

        for sig in signals:
            source = sig.get('source', 'Unknown')
            weight = self.strategy_weights.get(source, 0.5)
            action = sig.get('action', 'HOLD')
            conf = sig.get('confidence', 0.0)
            
            # Apply age-based decay if signal has a timestamp
            signal_age = sig.get('age', 0)  # age in ticks/bars
            if signal_age > 0:
                decay = max(0.5, 1.0 - signal_age * 0.05)  # 5% decay per tick, min 50%
                conf *= decay
            
            direction = 0
            if "LONG" in action or "BUY" in action:
                direction = 1
            elif "SHORT" in action or "SELL" in action:
                direction = -1
            
            score = direction * conf * weight
            vote_score += score
            total_weight += weight
            reasons.append(f"{source}:{action}")

        denom = max(total_weight, 1.0)
        final_score = vote_score / denom
        
        # Entry decision
        if final_score > self.min_confidence:
            self.decision_count += 1
            decision = {
                "action": "ENTRY_LONG", 
                "size": min(abs(final_score), 1.0),
                "reason": f"Long consensus: {reasons} (Score: {final_score:.2f})",
                "decision_id": self.decision_count
            }
            self.last_decision = decision
            self._record_decision(decision)
            return decision
        elif final_score < -self.min_confidence:
            self.decision_count += 1
            decision = {
                "action": "ENTRY_SHORT", 
                "size": min(abs(final_score), 1.0),
                "reason": f"Short consensus: {reasons} (Score: {final_score:.2f})",
                "decision_id": self.decision_count
            }
            self.last_decision = decision
            self._record_decision(decision)
            return decision
             
        return None

    def _record_decision(self, decision: Dict):
        """Record a decision to history, trimming to max_history."""
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

    def get_decision_history(self) -> List[Dict]:
        """Return the decision history list."""
        return self.decision_history

    def reset(self):
        """Clear position and decision history."""
        self.current_position = None
        self.last_decision = None
        self.decision_count = 0
        self.decision_history = []

