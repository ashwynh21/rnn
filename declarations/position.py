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

    def __init__(self, action: Action, state: State, nexter: State, price: float, balance: float, volume: float, sl: float, tp: float):
        self.nexter = nexter
        self.action = action
        self.state = state
        self.balance = balance
        self.price = price
        self.volume = volume

        self.elapsed = 0
        self.bias = [action.action] * 2
        self.sl = sl
        self.tp = tp

    def stoppedout(self, price: float) -> bool:
        return (price < self.sl) if self.action.action == 0 else (price > self.sl)

    def takeprofit(self, price: float) -> bool:
        return (price > self.tp) if self.action.action == 0 else (price < self.tp)
