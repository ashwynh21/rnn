"""
We define this class because we need an entity that will be able to measure the performance of our model.
So we define this class Metric that should be able to record the necessary data and run print outs for us
to gauge the performance.
"""
from typing import List

from declarations.action import Action


class Metric(object):
    # so we keep a list of how long our model survives for before losing in an iteration.
    longevity: List[int]
    # we keep the number of times the agent restarted.
    restarts: int
    # and then we keep the metric data from our loss and optimizer functions from the agent we are using.
    loss: List[float]
    # we keep the max profit here.
    best: float
    # we keep a metric for session profits
    profit: List[float]
    # actions stored here.
    actions: List[Action]

    """
    First we will run up with a function that will measure the number of accounts that the model is demolishing.
    """
    def __init__(self):
        self.restarts = 1
        self.loss = []
        self.longevity = []
        self.actions = []
        self.profit = []

    """
    The first function that we want to define is to be able to determine how the agent is surviving as a moving
    average on each iteration of the given data from the environment.
    """
    def survival(self, life: int) -> List[float]:
        # we simply append the value to the longevity list.
        self.longevity.append(life)

        # then compute the moving average and return it.
        size = 10
        i = 0

        ma = []
        while i < len(self.longevity) - size + 1:
            window = self.longevity[i: i + size]
            average = sum(window) / len(window)
            ma.append(average)

            i = i + 1

        return ma

    """
    After computing the survival of the agent we also need to find out how many times it had to start over before
    finishing one dataset.
    """
    def restart(self) -> int:
        self.restarts = self.restarts + 1

        return self.restarts

    """
    We define a function to update the loss that we are storing since we may want to plot the loss
    """
    def addloss(self, loss: float) -> List[float]:
        self.loss.append(loss)
        return self.loss

    """
    We define a function to compute the average survival rate of the agent.
    """
    def averagesurvival(self) -> float:
        return sum(self.longevity) / len(self.longevity)

    """
    We define a function that will allow to compute a session profit, since our model is now able to complete annual
    trading sessions.
    """
    def addprofit(self, profit: float):
        # we need to define this function to simply add the value provided to array profit.
        self.profit.append(profit)

    """
    we now define a function that will allow us to keep track of the max profit in our array list of profits.
    """
    def maxim(self, profit: float):
        if profit > self.best:
            self.best = profit

    """
    we define a metric function that will count the number of different actions taken in the environment.
    """
    def countaction(self, action: Action):
        self.actions.append(action)

    def reset(self):
        self.restarts = 1
        self.profit.clear()
        self.actions.clear()
        self.loss.clear()

    """
    We define a function that will count the number of random actions against the actual approximated actions.
    """
    def approximations(self) -> dict:
        def fill(action: Action, r: bool) -> bool:
            return action.random == r

        return {
            'random': len(list(filter(lambda a: fill(a, True), self.actions))),
            'predicted': len(list(filter(lambda a: fill(a, False), self.actions)))
        }

    """
    We now need a function to observe and summarize the actions that are taken by the agent so here we go.
    """
    def actionsummary(self):
        return {
            'buy': len(list(filter(lambda a: a.action == 0, self.actions))),
            'sell': len(list(filter(lambda a: a.action == 1, self.actions))),
            'hold': len(list(filter(lambda a: a.action == 2, self.actions))),
        }

    """
    We are going to need a function that will keep track of the maximum profit an account has made before being lost.
    """
