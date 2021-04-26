"""
We define this account class because we require a defined sort of score board for our agent that will allow it to
gauge quantitatively its performance.
"""
from typing import Dict

from declarations.action import Action
from declarations.position import Position
from declarations.result import Result
from declarations.state import State


class Account(object):
    balance: float
    risk: float

    ledger: Dict[str, Result]
    positions: Dict[str, Position]

    def __init__(self, balance):
        self.balance = balance
        self.positions = {}
        self.ledger = {}

        self.risk = 0.02

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
    
    We are going to redesign our closing function to better incorporate other factors of the position, the price of the
    market and other factors that would be relevant to the position.
    """

    def closable(self, state: State, action: Action) -> Dict[str, Position]:
        """
        We should only need the state of the environment since all the values to evaluate the validity of the
        position are based on computations between the two properties.
        :param action:
        :param state:
        :return:
        """
        data = {}

        for k, v in self.positions.items():
            if action.action != 2:
                v.bias.pop(0)
                v.bias.append(action.action)

            # so once we have the profit we need to get the next action from the agent, so we add it to the args list
            # so if the position is in the same direction as the action we hold otherwise we close.
            # this decision.
            # we add the condition that if the position bias is less than 0, then we close the position.

            # len(list(filter(lambda b: b != v.action.action, v.bias))) >= 2 or
            close = v.elapsed > 120 or v.stoppedout(state.price()) or v.takeprofit(state.price())

            if close:
                data[k] = v
            else:
                v.elapsed = v.elapsed + 1
        return data

    def reset(self, balance: float):
        self.balance = balance

    """
    We are going to need a function to calculate the risk that the account can manage before opening a position.
    """

    def stoploss(self) -> float:
        return self.balance * self.risk
