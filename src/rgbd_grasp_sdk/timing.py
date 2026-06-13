from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class TimingRecorder:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings[name] = perf_counter() - start
