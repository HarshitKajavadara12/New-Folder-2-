"""
GIGA SYSTEM - Alerting & Notifications
========================================

Send real-time trade alerts via Telegram and Discord.

Configuration is via environment variables:
    TELEGRAM_BOT_TOKEN   — Bot token from @BotFather
    TELEGRAM_CHAT_ID     — Chat / channel ID to send to
    DISCORD_WEBHOOK_URL  — Discord channel webhook URL

Usage
-----
    from utils.alerting import AlertManager

    alerts = AlertManager()
    alerts.send("BUY 0.5 BTC @ $65,000", level="trade")
    alerts.send_trade_alert("BUY", "BTCUSDT", 0.5, 65000)
    alerts.send_risk_alert("Daily loss limit 80% consumed")
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore


class AlertManager:
    """
    Multi-channel alerting: Telegram + Discord.

    Falls back gracefully if credentials are missing or libraries unavailable.
    All sends are non-blocking (fire-and-forget in a daemon thread).
    """

    LEVELS = ("info", "trade", "risk", "critical")

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        min_level: str = "trade",
    ):
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
        self._min_level_idx = self.LEVELS.index(min_level) if min_level in self.LEVELS else 0
        self._enabled = bool(_requests)

        channels = []
        if self.telegram_token and self.telegram_chat_id:
            channels.append("Telegram")
        if self.discord_webhook:
            channels.append("Discord")
        if channels:
            logger.info(f"[ALERT] Alerting active on: {', '.join(channels)}")
        else:
            logger.warning("[ALERT] No alert channels configured (set env vars)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str, level: str = "info") -> None:
        """
        Send an alert asynchronously if *level* meets the minimum threshold.
        """
        lvl_idx = self.LEVELS.index(level) if level in self.LEVELS else 0
        if lvl_idx < self._min_level_idx:
            return
        # Fire-and-forget
        t = threading.Thread(target=self._dispatch, args=(message, level), daemon=True)
        t.start()

    def send_trade_alert(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        extra: str = "",
    ) -> None:
        """Format and send a trade execution alert."""
        emoji = "\U0001f7e2" if side.upper() == "BUY" else "\U0001f534"
        msg = (
            f"{emoji} **{side.upper()}** {quantity} {symbol} @ ${price:,.2f}"
            f"\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        if extra:
            msg += f"\n{extra}"
        self.send(msg, level="trade")

    def send_risk_alert(self, message: str) -> None:
        """Send a risk-level alert."""
        msg = f"\u26a0\ufe0f **RISK ALERT**\n{message}\nTime: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        self.send(msg, level="risk")

    def send_critical_alert(self, message: str) -> None:
        """Send a critical-level alert (kill switch, connectivity loss, etc.)."""
        msg = f"\U0001f6a8 **CRITICAL**\n{message}\nTime: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        self.send(msg, level="critical")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(self, message: str, level: str) -> None:
        if not self._enabled:
            logger.debug(f"[ALERT] (dry-run) [{level}] {message}")
            return
        self._send_telegram(message)
        self._send_discord(message)

    def _send_telegram(self, message: str) -> None:
        if not (self.telegram_token and self.telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            resp = _requests.post(url, json=payload, timeout=10)  # type: ignore[union-attr]
            if resp.status_code != 200:
                logger.warning(f"[ALERT] Telegram send failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.error(f"[ALERT] Telegram error: {e}")

    def _send_discord(self, message: str) -> None:
        if not self.discord_webhook:
            return
        # Discord uses "content" field; convert Markdown bold (**) to Discord format (same)
        payload = {"content": message}
        try:
            resp = _requests.post(self.discord_webhook, json=payload, timeout=10)  # type: ignore[union-attr]
            if resp.status_code not in (200, 204):
                logger.warning(f"[ALERT] Discord send failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.error(f"[ALERT] Discord error: {e}")
