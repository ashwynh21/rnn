
"""
We define the class Action to allow us to abstract functions into the action made by our models.
"""
from typing import List

import numpy as np


class Action(object):
    probabilities: List[List[List[List[float]]]]
    random: bool

    def __init__(self, action: List[List[List[List[float]]]], random=True):
        self.probabilities = action
        self.action = np.argmax(action)
        self.random = random
