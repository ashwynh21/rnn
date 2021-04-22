from declarations import Account
from declarations.config import config
from declarations.agent import Agent
from declarations.environment import Environment

from declarations.metric import Metric
from declarations.util import train, test

if __name__ == '__main__':
    # init an agent, environment, and the account that the agent will be trading on the environment.
    account = Account(100)
    metric = Metric()

    environment = Environment('assets/NAS100.csv')
    agent = Agent('nasdaq', 128)
    # run our training function
    while metric.restarts != 0:
        train(environment, agent, account, metric)

        ax = metric.actionsummary()

        print(f'+----------------------------------------------------------+')
        print(f'| Training Session Summary on Agent                        |')
        print(f'+----------------------------------------------------------+')
        print(f'| Restarts        | {str(metric.restarts).ljust(39)}|')
        print(f'+----------------------------------------------------------+')
        print(f'| Actions Summary | {str(len(metric.actions)).ljust(39)}|')
        print(f'+----------------------------------------------------------+')
        print(f'| Buy |    {str(ax["buy"]).ljust(4)}    | Sell |    {str(ax["sell"]).ljust(4)}    | Hold |    {str(ax["hold"]).ljust(4)}    |')
        print(f'+----------------------------------------------------------+\n')

        environment.reset()
        metric.reset()

    # evaluate = Environment('assets/NAS100_2021.csv')
    # master = Agent('ustec', 128, evaluate=True)
    # master.load()
    # run our testing function
    # test(evaluate, master, account, metric)
