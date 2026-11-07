"""Wall-clock timing of training calls."""

import time


class Timer:
    """Context manager: `with Timer() as t: ...`, then read `t.seconds`."""

    def __init__(self):
        self.seconds = None
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
