import random
from typing import List, Optional


class ReservoirSamplerK:
    """
    Keep k uniform random samples from a stream of unknown length.
    Memory: O(k)
    Update: O(1)
    """

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.count = 0
        self.reservoir: List[float] = []

    def add(self, x):
        self.count += 1

        # Fill the reservoir first
        if len(self.reservoir) < self.k:
            self.reservoir.append(x)
            return

        # Replace an existing item with probability k / count
        j = random.randint(1, self.count)  # inclusive
        if j <= self.k:
            self.reservoir[j - 1] = x

    def sample(self) -> List[float]:
        return list(self.reservoir)

    def __len__(self) -> int:
        return len(self.reservoir)


class RunningStats:
    """
    Streaming mean and variance using Welford's algorithm.
    Memory: O(1)
    Update: O(1)

    variance() returns sample variance by default.
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def add(self, x: float):
        self.n += 1

        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def mean(self) -> float:
        return self._mean

    def variance(self, sample: bool = True) -> float:
        if self.n < 2:
            return 0.0
        if sample:
            return self._M2 / (self.n - 1)
        return self._M2 / self.n

    def std(self, sample: bool = True) -> float:
        return self.variance(sample=sample) ** 0.5

    def count(self) -> int:
        return self.n


if __name__ == "__main__":
    # -------------------------
    # Reservoir sampling (k)
    # -------------------------
    rs = ReservoirSamplerK(k=3)
    stream = [10, 20, 30, 40, 50, 60, 70]

    for x in stream:
        rs.add(x)

    print("Reservoir sample:", rs.sample())

    # -------------------------
    # Streaming statistics
    # -------------------------
    stats = RunningStats()
    values = [1, 3, 5]

    for x in values:
        stats.add(x)

    print("Count:", stats.count())
    print("Mean:", stats.mean())                 # 3.0
    print("Sample variance:", stats.variance()) # 4.0
    print("Population variance:", stats.variance(sample=False))  # 8/3
    print("Sample std:", stats.std())