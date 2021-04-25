from typing import List

from declarations.action import Action
from declarations.state import State


class Position(object):
    volume: float
    action: Action
    balance: float
    price: float
    state: State
    nexter: State
    # we add a property that will allow us to keep track of the bias toward the position
    bias: List[float]

    elapsed: int

    def __init__(self, action: Action, state: State, balance: float, price: float, volume: float, nexter: State):
        self.nexter = nexter
        self.action = action
        self.state = state
        self.balance = balance
        self.price = price
        self.volume = volume

        self.elapsed = 0
        self.bias = [action.action] * 2
