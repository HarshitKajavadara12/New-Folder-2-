"""
GIGA SYSTEM - Observer
======================

The Observer Realm is the "Eyes" of the system.
It records, measures, and visualizes the flow of information between Research, Reducer, and Live.

Principles:
- Read-Only: Never modifies state in other realms.
- Air-Gap compliant: Receives artifacts, does not create orders.
- Persistence: Logs are the source of truth for compliance.
"""

import logging
import logging.handlers
import time
import json
import threading
import queue
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class Observer:
    """
    The Witness (Phase 9 Optimized).
    Tracks signals, executions, and operations.
    NON-BLOCKING ASYNC LOGGING with log rotation.
    """
    
    # Log rotation settings
    MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per log file
    MAX_LOG_BACKUPS = 5  # Keep 5 rotated files (events.log.1, .2, etc.)
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("giga_system.observer")
        
        # Async Queue
        self.log_queue = queue.Queue(maxsize=10000)
        self.running = True
        self._write_lock = threading.Lock()  # Thread-safe file writes
        
        # Persistence paths
        self.state_path = Path("observer") / "state.json"
        self.events_path = Path("observer") / "events.log"
        # Ensure directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up rotating file handler for structured event logging
        self._events_handler = logging.handlers.RotatingFileHandler(
            str(self.events_path),
            maxBytes=self.MAX_LOG_SIZE_BYTES,
            backupCount=self.MAX_LOG_BACKUPS,
            encoding='utf-8'
        )
        self._events_handler.setFormatter(logging.Formatter('%(message)s'))
        self._events_logger = logging.getLogger("giga_system.observer.events")
        self._events_logger.addHandler(self._events_handler)
        self._events_logger.setLevel(logging.INFO)
        self._events_logger.propagate = False  # Don't duplicate to parent
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # Ensure we have a handler if not already configured via root logger
        if not self.logger.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s | OBSERVER | %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
        
        self.logger.setLevel(logging.INFO)
        
        self.metrics = {
            "total_signals": 0,
            "executed_signals": 0,
            "cumulative_pnl": 0.0,
            "avg_latency_ms": 0.0,
            "latency_measurements": 0
        }
        # Persistence paths
        self.state_path = Path("observer") / "state.json"
        self.events_path = Path("observer") / "events.log"
        # Ensure directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Fire and forget."""
        try:
            self.log_queue.put_nowait({
                "ts": time.time(),
                "type": event_type,
                "data": data
            })
        except queue.Full:
            pass # Drop logs if overloaded (Phase 9 Rule)

    def _worker(self):
        """Background writer using rotating log handler."""
        while self.running:
            try:
                item = self.log_queue.get(timeout=1.0)
                # Write via rotating logger (auto-rotates at MAX_LOG_SIZE_BYTES)
                self._events_logger.info(json.dumps(item))
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Observer worker error: {e}")

    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        if not self.state_path.exists():
            with open(self.state_path, "w") as f:
                json.dump(self.metrics, f)

    def log_signal(self, signal_artifact: Dict[str, Any]):
        """
        Log a decision made by the Reducer.
        """
        self.metrics["total_signals"] += 1
        
        # Log structure for parsing later
        msg = (
            f"SIGNAL RECEIPT | "
            f"Signal: {signal_artifact.get('signal')} | "
            f"Sym: {signal_artifact.get('symbol')} | "
            f"Size: {signal_artifact.get('lot_size')} | "
            f"Conf: {signal_artifact.get('confidence'):.2f}"
        )
        self.logger.info(msg)
        # Write event via rotating logger (thread-safe, auto-rotated)
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": "signal",
                "signal": signal_artifact.get('signal'),
                "symbol": signal_artifact.get('symbol'),
                "size": signal_artifact.get('lot_size'),
                "confidence": signal_artifact.get('confidence')
            }
            self._events_logger.info(json.dumps(event))
            # Persist metrics
            with self._write_lock:
                with open(self.state_path, "w") as sf:
                    json.dump(self.metrics, sf)
        except Exception:
            pass

    def log_execution(self, 
                      signal: str, 
                      executed: bool, 
                      price: float, 
                      size: int,
                      pnl: float = 0.0, 
                      latency_ms: float = 0.0):
        """
        Log an execution outcome from the Live Realm.
        """
        if executed:
            self.metrics["executed_signals"] += 1
            self.metrics["cumulative_pnl"] += pnl
            
            # Update average latency
            n = self.metrics["latency_measurements"]
            current_avg = self.metrics["avg_latency_ms"]
            self.metrics["avg_latency_ms"] = (current_avg * n + latency_ms) / (n + 1)
            self.metrics["latency_measurements"] += 1

            msg = (
                f"EXECUTION CONFIRMED | "
                f"Action: {signal} | "
                f"Price: {price} | "
                f"Size: {size} | "
                f"PnL: {pnl:.2f} | "
                f"Latency: {latency_ms:.1f}ms"
            )
            self.logger.info(msg)
            # Write event via rotating logger (thread-safe, auto-rotated)
            try:
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "execution",
                    "action": signal,
                    "price": price,
                    "size": size,
                    "pnl": pnl,
                    "latency_ms": latency_ms
                }
                self._events_logger.info(json.dumps(event))
                with self._write_lock:
                    with open(self.state_path, "w") as sf:
                        json.dump(self.metrics, sf)
            except Exception:
                pass
        else:
            self.logger.warning(f"EXECUTION FAILED/SKIPPED | Signal: {signal}")

    def report(self) -> Dict[str, Any]:
        """
        Return current session metrics.
        """
        self.logger.info("--- OBSERVER SESSION REPORT ---")
        self.logger.info(f"Total Signals: {self.metrics['total_signals']}")
        self.logger.info(f"Executed:      {self.metrics['executed_signals']}")
        self.logger.info(f"Cum PnL:       ${self.metrics['cumulative_pnl']:.2f}")
        self.logger.info(f"Avg Latency:   {self.metrics['avg_latency_ms']:.2f}ms")
        # Persist final state
        try:
            with open(self.state_path, "w") as sf:
                json.dump(self.metrics, sf)
        except Exception:
            pass
        return self.metrics
