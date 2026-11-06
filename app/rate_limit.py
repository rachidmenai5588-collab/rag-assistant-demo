"""In-memory token bucket rate limiter — ARCH-02 §9.

30 requests/minute per IP (config), plus the daily cap in server.py.
This bounds the buyer's OpenAI bill if the widget is scraped. It is
not a WAF and does not claim to be one.
"""

import time


class TokenBucket:
    def __init__(self, capacity, window_seconds):
        self.capacity = capacity
        self.window = window_seconds
        self.tokens = float(capacity)
        self.last = time.monotonic()

    def take(self, n=1):
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity,
                          self.tokens + elapsed * self.capacity / self.window)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
