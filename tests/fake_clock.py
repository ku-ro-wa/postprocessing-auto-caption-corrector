"""A clock tests set by hand, for code that takes a ``clock`` callable."""

from __future__ import annotations

from datetime import datetime


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now
