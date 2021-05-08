"""
We define this class to manage a single state for the application, we will use this in the envrionment system.
so that we do not need to worry about working with the primitive types
"""
from typing import List
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt


scalar = MinMaxScaler()


class State(object):
    def __init__(self, data: List[List[float]]):
        self.data = data

    def normalize(self) -> np.array:
        return self.__normalize__()

    def price(self) -> float:
        return self.data[119][4]

    def spread(self) -> float:
        return self.data[119][5]

    def __normalize__(self) -> np.array:
        # first we need to remove the first element of each element in the List
        cache = []
        time = []

        # we iterate through 12 elements and move each first element within the array
        for i in range(len(self.data)):
            time.append(self.data[i][0])
            cache.append(self.data[i][1:])

        # we will iterate through the data and normalize each
        scaled = scalar.fit_transform(cache).tolist()
        for i, candle in enumerate(scaled):
            candle.append(time[i])

        return np.array([np.array(scaled).reshape(120, 6, 1)])

    # we define a simple function that will plot the OHLC gotten from the above functions
    def plot(self, name: str):
        y = self.normalize()[0]
        x = [x[5] for x in y]

        plt.plot(x, [x[0] for x in y], label=f'{name} - open')
        plt.plot(x, [x[1] for x in y], label=f'{name} - high')
        plt.plot(x, [x[2] for x in y], label=f'{name} - low')
        plt.plot(x, [x[3] for x in y], label=f'{name} - close')
        plt.legend()
        plt.show()
