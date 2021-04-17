
"""
We define the class Action to allow us to abstract functions into the action made by our models.
"""
from typing import List

import numpy as np


class Action(object):
    probabilities: List[List[float]]

    def __init__(self, action: List[List[float]]):
        self.probabilities = action
        self.action = np.argmax(action)
