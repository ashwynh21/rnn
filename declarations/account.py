"""
We define this account class because we require a defined sort of score board for our agent that will allow it to
gauge quantitatively its performance.
"""
from typing import Dict
from declarations.position import Position
from declarations.result import Result


class Account(object):
    balance: float

    ledger: Dict[str, Result]
    positions: Dict[str, Position]

    def __init__(self, balance):
        self.balance = balance
        self.positions = {}
        self.ledger = {}

    """
    Since our definition here is purely score boarding we need functions to update the properties that we have defined.
    
    So the account will also have a rule for the way it handles positions because right now our algorithm for closing
    a position is not yet well defined so we will for now opt with defining a structured closing strategy.
    """
    def record(self, k: str, v: Position):
        """
        A function that will allow to open a position...
        :return:
        """
        self.balance = v.balance
        self.positions[k] = v

    def archive(self, k, v: Result):
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
        return self.balance > (volume * price / 100)

    """
    Now we define a function that will get the closable positions as a dictionary
    """
    def closable(self) -> Dict[str, Position]:
        data = {}
        """
        We need to devise a better way to close positions because the current strategy is too difficult for the agent
        to work with. We require a more dynamic method to which we can close positions.
        
        1. Let us consider the data that we would like to use in considering whether a position is a valid close.
            i. 
        """
        for k, v in self.positions.items():
            # so this is our closing strategy, of course in the future we will optimize and allow the agent to act on
            # this decision.
            if v.elapsed >= 2:
                data[k] = v
            else:
                v.elapsed = v.elapsed + 1

        return data

    def reset(self, balance: float):
        self.balance = balance
