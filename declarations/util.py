"""
We are certainly going to require a set of functions that are going to help us run any conversions that we require.
"""
# first we are going to need a function that will convert our input data set to a normalized format.
from declarations import Environment, Agent, Account, Metric, Experience

"""
the training function defined below only runs through one iteration of the environment without resetting anything.

the goal in this function is to get the agent to survive the environment for one iteration.
"""

