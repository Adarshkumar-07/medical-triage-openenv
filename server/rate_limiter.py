"""
server/rate_limiter.py

Token-bucket rate limiter — per-client, in-process, thread-safe.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def consume(self, cost: float = 1.0) -> Tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        deficit = cost - self.tokens
        return False, round(deficit / self.refill_rate, 2)


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_multiplier: float = 1.5,
        enabled: bool = True,
    ) -> None:
        self._rpm = requests_per_minute
        self._refill_rate = requests_per_minute / 60.0
        self._capacity = requests_per_minute * burst_multiplier
        self._enabled = enabled
        self._buckets: Dict[str, TokenBucket] = defaultdict(self._new_bucket)
        self._lock = threading.Lock()

    def check(self, client_key: str, cost: float = 1.0) -> Tuple[bool, float]:
        if not self._enabled:
            return True, 0.0
        with self._lock:
            return self._buckets[client_key].consume(cost)

    def reset(self, client_key: str) -> None:
        with self._lock:
            self._buckets.pop(client_key, None)

    def client_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _new_bucket(self) -> TokenBucket:
        return TokenBucket(capacity=self._capacity, refill_rate=self._refill_rate)