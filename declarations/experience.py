"""
We define the class experience so that we are able to represent the experience data that we have collected for our
memory usage.
"""
from typing import List, NamedTuple


class Experience(NamedTuple):
    state: List[List[float]]
    action: int
    reward: float
    next: List[List[float]]
