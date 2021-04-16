"""
We define this account class because we require a defined sort of score board for our agent that will allow it to
gauge quantitatively its performance.
"""
from collections import deque
from typing import Dict
from declarations.position import Position
from declarations.result import Result


class Account(object):
    balance: float

    ledger: Dict[str, Result]
    positions: Dict[str, Position]

    def __init__(self, balance):
        self.balance = balance

    """
    Since our definition here is purely score boarding we need functions to update the properties that we have defined.
    
    So the account will also have a rule for the way it handles positions because right now our algorithm for closing
    a position is not yet well defined so we will for now opt with defining a structured closing strategy.
    """
    def open(self, k: str, v: Position):
        """
        A function that will allow to open a position...
        :return:
        """
        self.balance = v.balance
        self.positions[k] = v

    def close(self, k, v: Result):
        """
        A function that will allow us to close one of the positions that we have in the positions property that we
        have defined.
        :return:
        """
        self.balance = self.balance + v.profit

        del self.positions[k]
        self.ledger[k] = v

    """
    Let us define a function that will allow us to check if the available balance is going to be enough to open a
    said position before submitting the request to the environment. If the balance is not enough then we raise an
    exception that will cause the session or episode to restart.
    """
    def isable(self, price: float, volume: float) -> bool:
        return self.balance < (volume * price / 100)
