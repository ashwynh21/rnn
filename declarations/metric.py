"""
We define this class because we need an entity that will be able to measure the performance of our model.
So we define this class Metric that should be able to record the necessary data and run print outs for us
to gauge the performance.
"""
from typing import List, Generator


class Metric(object):
    # so we keep a list of how long our model survives for before losing in an iteration.
    longevity: List[int]
    # we keep the number of times the agent restarted.
    restarts: int
    # and then we keep the metric data from our loss and optimizer functions from the agent we are using.
    loss: List[float]

    """
    First we will run up with a function that will measure the number of accounts that the model is demolishing.
    """
    def __init__(self):
        self.restarts = 0
        self.longevity = []

    """
    The first function that we want to define is to be able to determine how the agent is surviving as a moving
    average on each iteration of the given data from the environment.
    """
    def survival(self, life: int) -> Generator[float]:
        # we simply append the value to the longevity list.
        self.longevity.append(life)

        # then compute the moving average and return it.
        size = 10
        i = 0
        while i < len(self.longevity) - size + 1:
            window = self.longevity[i: i + size]
            average = sum(window) / len(window)
            i = i + 1
            yield average

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
