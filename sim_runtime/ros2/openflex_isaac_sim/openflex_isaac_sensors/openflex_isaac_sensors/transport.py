"""Bounded latest-sample transport primitives."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedFrameQueue(Generic[T]):
    """Thread-safe bounded queue that drops the oldest item when full."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: deque[T] = deque(maxlen=maxsize)
        self._lock = Lock()
        self.put_count = 0
        self.drop_count = 0

    def put(self, item: T) -> None:
        with self._lock:
            self.put_count += 1
            if len(self._items) == self.maxsize:
                self._items.popleft()
                self.drop_count += 1
            self._items.append(item)

    def get(self) -> T | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def drain(self) -> list[T]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "capacity": self.maxsize,
                "depth": len(self._items),
                "put_count": self.put_count,
                "drop_count": self.drop_count,
            }
