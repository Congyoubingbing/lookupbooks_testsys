from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None

@dataclass
class TaskSession:
    """A single checked-out API session.

    One book pipeline MUST occupy exactly one key until the pipeline finishes.
    We attach retry policy here so all downstream calls share the same knobs.
    """

    api_key: str
    base_url: str
    timeout_s: int
    client: Any
    max_retries: int = 0
    backoff_s: List[float] | None = None

class ApiPool:
    """
    Key-per-pipeline pool:
    - One book pipeline borrows one key for the entire processing.
    - Keys are returned only when the pipeline completes.
    """
    def __init__(
        self,
        keys: List[str],
        base_url: str,
        timeout_s: int = 180,
        *,
        max_retries: int = 0,
        backoff_s: Optional[List[float]] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Please `pip install -r requirements.txt`.")
        if not keys:
            raise ValueError("No API keys found. Set QWEN_API_KEY_0..9 (or QWEN_API_KEY_1..10) and rebuild the pool.")
        self.keys = keys[:]
        self.base_url = base_url
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_s = list(backoff_s) if backoff_s else [2.0, 4.0, 8.0]
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._available = keys[:]

    @contextmanager
    def session(self) -> TaskSession:
        api_key = None
        with self._cv:
            while not self._available:
                self._cv.wait(timeout=0.2)
            api_key = self._available.pop(0)
        try:
            client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout_s)
            yield TaskSession(
                api_key=api_key,
                base_url=self.base_url,
                timeout_s=self.timeout_s,
                client=client,
                max_retries=self.max_retries,
                backoff_s=self.backoff_s,
            )
        finally:
            with self._cv:
                self._available.append(api_key)
                self._cv.notify()
