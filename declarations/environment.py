"""
In this file we are going to be modelling our environment so that we are able to handle the learning data properly and
also make it easy for us to query for information that can be used by our application agent.
"""
from datetime import datetime, timedelta
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from declarations.action import Action
from declarations.position import Position
from declarations.result import Result
from declarations.state import State

scalar = MinMaxScaler()


class Environment:
    pair: str
    """
    To accommodate the cases of
    """
    def __init__(self, training: str, pair: str):
        # we define the initial time step state to zero
        self.pair = pair

        self.__step = 0
        """
        So here we will define the input data set, from the a file path to the CSV learning and testing data.
        """
        self.__data = pd.read_csv(training)
        # we then need to keep a nice digested version of the information in a way that will allow to cross reference
        # the information between both data sets.
        self.__pdata = Environment.parse(self.__data)
        # okay that's all done, we now need the environment to be able to handle the actions that an agent would take,
        # since it is the environment that defines what actions are taken on it. we define the actions here.

    def __next__(self):
        self.__step = self.__step + 1

    def step(self) -> (int, Union[State, None]):
        """
        We define the function step that will be the environment progression through the given data. So what we expect
        here is that the function will check some iteration value from its own controlled state system so that is can
        return the necessary data that is relevant to the given time step that it is in.
        :return:
        """
        return self.__step, self.__pdata[self.__step]

    def reset(self):
        self.__step = 0

    """
    We now need to define the functions that will perform market executions on the defined stepping environment
    structure that we have so let us begin by defining the function that will allow to place market orders...
    """
    def buy(self, volume: float, balance: float) -> Position:
        """
        we will separate the buy sell and hold functions fairly to make our code more understandable regardless of the
        redundancy. The function should return a sort of receipt for the agent so that they are able to record the
        positions that has been generated. Note that the reward of the action will not be computed just yet since the
        environment must progress to the next state.

        action: the action will buy, sell, or hold.
        volume: the volume is the amount of the given data we are acting on.
        balance: we will need the balance to check if the action is even possible.
        :return:
        """
        # need to also check if order is possible
        # so since to get the environment state the step function is called, then acted on so the call done to act,
        # will be on the previous step since the step is immediately incremented after step is called
        cost = self.__pdata[self.__step].price()
        spread = self.__pdata[self.__step].spread()

        price = cost + spread
        state = self.__pdata[self.__step]

        if balance < (volume * price / 100):
            raise Exception('Insufficient balance to buy')

        # then we update the time step...
        self.__next__()

        print(f'[BUY]: {balance}, {volume}, {-self.getprofit(self.pipstoprofit(50), volume)}', price, (volume * price / 100))
        # we then need to return the data that the agent will need to update its own position in the environment.
        return Position(
            action=Action([[[[1, 0, 0]]]]),
            volume=volume,
            balance=balance - (volume * price / 100),
            state=state,
            nexter=self.__pdata[self.__step + 1 if self.__step < len(self.__pdata) - 1 else -1],
            price=price,
            sl=(price - self.pipstoprofit(50)),
            tp=(price + self.pipstoprofit(250)),
        )

    def sell(self, volume: float, balance: float) -> Position:
        cost = self.__pdata[self.__step].price()
        spread = self.__pdata[self.__step].spread()

        price = cost - spread
        state = self.__pdata[self.__step]

        if balance < (volume * price / 100):
            raise Exception('Insufficient balance to buy')

        # then we update the time step...
        self.__next__()

        # we then need to return the data that the agent will need to update its own position in the environment.
        print(f'[SELL]: {balance}, {volume}, {-self.getprofit(self.pipstoprofit(50), volume)}', price, (volume * price / 100))

        return Position(
            action=Action([[[[0, 1, 0]]]]),
            state=state,
            nexter=self.__pdata[self.__step + 1 if self.__step < len(self.__pdata) - 1 else -1],
            volume=volume,
            balance=balance - (volume * price / 100),
            price=price,
            sl=(price + self.pipstoprofit(50)),
            tp=(price - self.pipstoprofit(250)),
        )

    def hold(self) -> Position:
        state = self.__pdata[self.__step]

        # then we update the time step...
        self.__next__()

        print(f'[HOLD]')
        return Position(
            action=Action([[[[0, 0, 1]]]]),
            state=state,
            balance=0,
            volume=0,
            nexter=self.__pdata[self.__step],
            price=0,
            sl=0,
            tp=0,
        )

    def close(self, position: Position) -> Result:
        """
        Now since is the closing component of an action we will need to use this function to compute the reward coming
        from the environment, that is, the profit or the loss. We will define a representation of the result called
        Result.

        When this function is called, we assume that the close of the provided position is called on the current price
        that has been returned by our step function. so then we will compute the price from the price and then we have.
        :param position:
        :return:
        """
        profit = 0

        if position.stoppedout(self.step()[1].price()):
            if position.action.action == 0:
                profit = position.sl - position.price
            elif position.action.action == 1:
                profit = position.price - position.sl
        elif position.takeprofit(self.step()[1].price()):
            if position.action.action == 0:
                profit = position.tp - position.price
            elif position.action.action == 1:
                profit = position.price - position.tp
        else:
            if position.action.action == 0:
                profit = self.step()[1].price() - position.price
            elif position.action.action == 1:
                profit = position.price - self.step()[1].price()

        # now we get the final result from the profit and how many units that were invested
        profit = self.getprofit(profit, position.volume)
        # in the result of closing the position we need to return the reward, and the state of the environment.
        return Result(profit=profit, state=self.step()[1])

    """
    Here we define a set of helper functions for the environment that will help convert the external data into something
    useful...
    """
    @staticmethod
    def track(data):
        t = 120
        for i in range(len(data) - t):
            yield data[i: i + t]

    @staticmethod
    def dater(date: datetime):
        # we determine the closing time of the market
        close = date + timedelta(days=(4 - date.weekday()))
        close = close.replace(hour=19, minute=0, second=0)

        # we determine the normal of the week
        normal = date - timedelta(days=date.weekday() + 1)
        normal = normal.replace(hour=22, minute=0, second=0)
        fit = np.array([normal.timestamp(), date.timestamp(), close.timestamp()]).reshape(-1, 1)

        # then we return the difference in seconds
        return scalar.fit_transform(fit)[1][0]

    @staticmethod
    def parse(data: pd.DataFrame) -> List[State]:
        # we remove the columns we do not need
        del data['tick_volume']
        del data['real_volume']

        # here we convert the string date to numbers of type float32
        data['time'] = pd.to_datetime(data['time'])
        # time relative to weeks
        data['time'] = data['time'].map(Environment.dater)

        # we then need to split the data into sets of 12
        data = data.to_numpy()
        data = list(Environment.track(data))

        # then we convert the list of lists to a list of state objects
        return list(map(lambda s: State(s), data))

    """
    we need to define a function that checks if the episode or state data is completed its iteration so that we are
    able to stop querying for data that isnt there yet.
    """
    def done(self) -> bool:
        return self.__step == len(self.__pdata) - 1

    def pipstoprofit(self, pips: float):
        pips = pips * 10000
        if 'JPY' not in self.pair:
            multiplier = 0.0001
        else:
            multiplier = 0.01

        return pips * multiplier

    """
    We define a function that will compute the profit of a given position.
    """
    def getprofit(self, profit: float, volume: float):
        if 'JPY' not in self.pair:
            multiplier = 0.0001
        else:
            multiplier = 0.01

        pips = profit / multiplier

        return pips * volume * multiplier
