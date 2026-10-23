import os
import time
import requests
from typing import Tuple, Optional

class TelegramNotifier:
    """
    Safe and robust Telegram Alert Dispatcher for SafeTransit-AI.
    Features:
    - Secret masking in all log output
    - Configurable cooldown window (default: 60s) to prevent spamming
    - Duplicate suppression for ongoing incidents
    - Strict network timeouts (5s) and graceful error handling
    - Test alert functionality
    - Explicit disclaimer: internal system alert, not direct 911/EMS dispatch
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, cooldown_seconds: int = 60):
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')
        try:
            self.cooldown_seconds = int(os.environ.get('ALERT_COOLDOWN_SECONDS', cooldown_seconds))
        except (ValueError, TypeError):
            self.cooldown_seconds = cooldown_seconds
        
        self.last_sent_time = 0.0
        self.last_severity = None
        self.sent_count = 0

    @property
    def is_configured(self) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        # Check against common dummy/template values
        invalid_tokens = ['your_', 'placeholder', 'token', 'actual_chat_id', 'your_actual_chat_id']
        token_lower = self.bot_token.lower()
        chat_lower = str(self.chat_id).lower()
        if any(inv in token_lower for inv in invalid_tokens):
            return False
        if any(inv in chat_lower for inv in invalid_tokens):
            return False
        return True

    def get_masked_token(self) -> str:
        if not self.bot_token:
            return "<UNSET>"
        if len(self.bot_token) <= 8:
            return "***"
        return f"{self.bot_token[:4]}...{self.bot_token[-4:]}"

    def get_masked_chat_id(self) -> str:
        if not self.chat_id:
            return "<UNSET>"
        chat_str = str(self.chat_id)
        if len(chat_str) <= 4:
            return "***"
        return f"{chat_str[:2]}...{chat_str[-2:]}"

    def send_incident_alert(
        self,
        severity: str = "MAJOR",
        timestamp_str: str = "",
        confidence: float = 0.0,
        snapshot_path: Optional[str] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Send an incident alert message if configured and not within cooldown.
        Returns: (success: bool, status_message: str)
        """
        now = time.time()
        elapsed = now - self.last_sent_time

        if not self.is_configured:
            msg = f"[TelegramNotifier] Skipped: Bot Token ({self.get_masked_token()}) or Chat ID ({self.get_masked_chat_id()}) not configured."
            print(msg)
            return False, msg

        if not force and elapsed < self.cooldown_seconds:
            msg = f"[TelegramNotifier] Cooldown active ({int(self.cooldown_seconds - elapsed)}s remaining). Suppressed duplicate alert."
            print(msg)
            return False, msg

        # Construct message with explicit notification scope
        msg_text = (
            f"🚨 *SafeTransit-AI Road Incident Detected*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Severity Estimate:* {severity.upper()}\n"
            f"• *Confidence:* {confidence * 100:.1f}%\n"
            f"• *Timestamp:* {timestamp_str or time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ *Notice:* Internal Telegram channel notification dispatched to security team. "
            f"This is not a direct dispatch to municipal 911 / EMS emergency services."
        )

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": msg_text,
                "parse_mode": "Markdown"
            }
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.last_sent_time = now
                self.last_severity = severity
                self.sent_count += 1
                success_msg = f"[TelegramNotifier] Alert successfully sent to Telegram chat {self.get_masked_chat_id()}."
                print(success_msg)
                return True, success_msg
            else:
                err_msg = f"[TelegramNotifier] Telegram API returned HTTP {resp.status_code}: {resp.text}"
                print(err_msg)
                return False, err_msg
        except Exception as e:
            err_msg = f"[TelegramNotifier] Network exception sending alert: {e}"
            print(err_msg)
            return False, err_msg

    def test_connection(self) -> Tuple[bool, str]:
        """
        Send a non-incident diagnostic test message to verify bot credentials.
        """
        if not self.is_configured:
            return False, f"Telegram credentials are incomplete. Token: {self.get_masked_token()}, Chat ID: {self.get_masked_chat_id()}"
        
        test_msg = (
            f"ℹ️ *SafeTransit-AI Diagnostic Test*\n"
            f"Telegram alert channel connectivity verified successfully.\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(url, json={"chat_id": self.chat_id, "text": test_msg, "parse_mode": "Markdown"}, timeout=5)
            if resp.status_code == 200:
                return True, f"Telegram test message successfully delivered to chat {self.get_masked_chat_id()}."
            else:
                return False, f"Telegram API error {resp.status_code}: {resp.text}"
        except Exception as e:
            return False, f"Connection failed: {e}"

# Global singleton instance
notifier = TelegramNotifier()
