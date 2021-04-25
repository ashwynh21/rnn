"""
We define the class Action to allow us to abstract functions into the action made by our models.
"""
from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()


class Action(object):
    probabilities: List[List[List[List[float]]]]
    random: bool

    def __init__(self, action: List[List[List[List[float]]]], random=True):
        self.probabilities = action
        self.action = np.argmax(action)
        self.random = random

    def normalize(self) -> np.array:
        return [[[scalar.fit_transform(np.array(self.probabilities[0][0][0]).reshape(-1, 1)).flatten().tolist()]]]
