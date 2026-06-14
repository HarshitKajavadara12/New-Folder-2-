"""
Minimal Observer Dashboard
Reads `observer/state.json` and `observer/events.log` and displays metrics and recent events.
Run with: `streamlit run visualization/observer_app.py`
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="GIGA Observer Dashboard", layout="wide")

STATE_PATH = Path("observer") / "state.json"
EVENTS_PATH = Path("observer") / "events.log"

st.title("GIGA Observer Dashboard")

col1, col2 = st.columns([2, 3])

with col1:
    st.header("Session Metrics")
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    else:
        metrics = None

    if metrics:
        st.metric("Total Signals", metrics.get("total_signals", 0))
        st.metric("Executed", metrics.get("executed_signals", 0))
        st.metric("Cumulative PnL", f"${metrics.get('cumulative_pnl', 0.0):.2f}")
        st.metric("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0.0):.2f}")
    else:
        st.info("No observer state found. Run the system to generate metrics.")

with col2:
    st.header("Recent Events")
    if EVENTS_PATH.exists():
        events = []
        try:
            with open(EVENTS_PATH, "r") as ef:
                for line in ef:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            events = []

        if events:
            # Show last 20 events
            recent = events[-20:][::-1]
            for ev in recent:
                ts = ev.get("timestamp")
                typ = ev.get("type")
                if typ == "signal":
                    st.write(f"[{ts}] SIGNAL {ev.get('signal')} {ev.get('symbol')} size={ev.get('size')} conf={ev.get('confidence')}")
                else:
                    st.write(f"[{ts}] EXEC {ev.get('action')} price={ev.get('price')} size={ev.get('size')} pnl={ev.get('pnl')}")
        else:
            st.info("No events recorded yet.")
    else:
        st.info("No events log found. Run the system to generate events.")

st.markdown("---")
st.markdown("This is a minimal read-only dashboard. It does not and will not modify live execution.")
