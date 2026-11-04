from collections import deque


class MovingAverage:
    def __init__(self, window: int):
        self._samples = deque[float](maxlen=window)

    def add(self, sample: float) -> None:
        self._samples.append(sample)

    def value(self) -> float | None:
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)
