from typing import List, NamedTuple


class Position(NamedTuple):
    volume: float
    action: int
    balance: float
    price: float
    state: List[List[float]]
