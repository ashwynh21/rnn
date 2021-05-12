
"""
In this class we are going to define functionality that will allow price level detection.
"""
from typing import List

from mplfinance.original_flavor import candlestick_ohlc
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.dates as dts
import pandas as pd
import numpy as np


class Level(object):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def plot(self, data: pd.DataFrame, highs: List[List[float]], lows: List[List[float]]):
        fig, ax = plt.subplots()

        ax.set_title(f'{self.symbol} - {data.iloc[0][0]}')

        candlestick_ohlc(ax, data.to_numpy(), width=0.005, colorup='green', colordown='red', alpha=1)

        for low in lows:
            ax.axhline(low[0], color='blue', ls='--')
        for high in highs:
            ax.axhline(high[0], color='orange', ls='--')

        ax.grid(True)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        ax.xaxis.set_major_formatter(dts.DateFormatter('%Y/%m/%d %H:%M'))
        fig.autofmt_xdate()

        fig.tight_layout()

        plt.show()

    @staticmethod
    def getclusters(data: pd.DataFrame, saturation_point=0.05):
        """
        :param data: dataframe
        :param saturation_point: The amount of difference we are willing to detect
        :return: clusters with optimum K centers
        This method uses elbow method to find the optimum number of K clusters
        We initialize different K-means with 1..10 centers and compare the inertias
        If the difference is no more than saturation_point, we choose that as K and move on
        """
        wcss = []
        k_models = []

        size = min(4, len(data.index))
        for i in range(1, size):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            k_models.append(kmeans)

        # Compare differences in inertias until it's no more than saturation_point
        optimum_k = len(wcss) - 1
        for i in range(0, len(wcss) - 1):
            diff = abs(wcss[i + 1] - wcss[i])
            if diff < saturation_point:
                optimum_k = i
                break

        optimum_clusters = k_models[optimum_k]

        return optimum_clusters

    @staticmethod
    def highlow(data: pd.DataFrame):
        lows = pd.DataFrame(data=data, index=data.index, columns=['low'])
        highs = pd.DataFrame(data=data, index=data.index, columns=['high'])

        low_clusters = Level.getclusters(lows)
        low_centers = low_clusters.cluster_centers_
        low_centers = np.sort(low_centers, axis=0)

        high_clusters = Level.getclusters(highs)
        high_centers = high_clusters.cluster_centers_
        high_centers = np.sort(high_centers, axis=0)

        """
        So we want to pair up the highs with their nearest lows
        """
        return [[h.tolist()[0], low_centers[(np.abs(low_centers - h)).argmin()].tolist()[0]] for h in high_centers]
