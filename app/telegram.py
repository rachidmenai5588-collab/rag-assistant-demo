"""Telegram integration — ARCH-02 §2.9, §7.2.

Direct Bot API calls via `requests` (no framework): the bot does two
things (receive a message, send a reply) plus escalation messages.
Webhook mode (not polling) — polling needs an always-on process, which
conflicts with Render free spin-down.
"""

import hmac

import requests

API_BASE = "https://api.telegram.org/bot"


def send_message(token, chat_id, text):
    """Send a message. Returns True on success. Raises on API failure."""
    resp = requests.post(
        f"{API_BASE}{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Telegram sendMessage failed: HTTP {resp.status_code}: "
            f"{resp.text[:200]}")
    return True


def verify_webhook_secret(request, secret):
    """Constant-time check of Telegram's X-Telegram-Bot-Api-Secret-Token."""
    given = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    return hmac.compare_digest(given, secret)


def send_escalation(token, owner_chat_id, question):
    """M16: forward a refused question to the owner (within 2 minutes
    of the refusal — this call is made synchronously on refusal)."""
    return send_message(
        token, owner_chat_id,
        f"سؤال لم يستطع البوت الإجابة عنه:\n{question}")
