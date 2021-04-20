from typing import NamedTuple

from declarations.state import State


class Result(NamedTuple):
    profit: float
    state: State

    """
    We define a function to compute the result of said profit.
    """
    def reward(self) -> float:
        """
        if the result is positive as in profit was generated, then we give the agent result 1 point,
        if the result was loss then the agent loses 1 point.
        :return:
        """
        return 1 if self.profit > 0 else -0.2
