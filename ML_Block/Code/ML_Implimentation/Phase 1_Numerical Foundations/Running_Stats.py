import numpy as np
from collections import  deque
class RunningStats:
    def __init__(self):
        self.n = 0
        self.Mean =0
        self.M = 0
    def add(self, x):
        self.n += 1
        self.delta1 = x-self.Mean
        self.Mean += (x-self.Mean) / self.n
        self.delta2 = x-self.Mean
        self.M += self.delta1 * self.delta2

    def mean(self):
        return self.Mean

    def variance(self):
        if self.n<2:
            return 0
        return self.M/(self.n -1)

class MovingAverage:
    def __init__(self, k):
        self.k = k
        self.que = deque()
    def add(self, x):
        if len(self.que)<self.k:
            self.que.append(x)
        else:
            self.que.popleft()
            self.que.append(x)
    def average(self):
        return np.mean(self.que)
if __name__ == "__main__":
    tracker = RunningStats()

    tracker.add(1)
    tracker.add(3)
    tracker.add(5)

    print("mean:", tracker.mean())
    print("variance:", tracker.variance())