"""
We define the class experience so that we are able to represent the experience data that we have collected for our
memory usage.
"""
from typing import NamedTuple

from declarations.action import Action
from declarations.state import State


class Experience(NamedTuple):
    state: State
    action: Action
    reward: float
    next: State
