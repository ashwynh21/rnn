from declarations import Account
from declarations.config import config
from declarations.agent import Agent
from declarations.environment import Environment
from tqdm import tqdm

if __name__ == '__main__':
    # init an agent, environment, and the account that the agent will be trading on the environment.
    agent = Agent('nasdaq')
    environment = Environment('assets/NAS100.csv', 'assets/NAS100_2021.csv')
    account = Account(0)

    """
    Now we need to devise a way that the agent will trade the account, by getting the state from the environment and
    acting on it using the account. Having made a choice on the action to take then we need to use the result action
    with the account.
    
    So we then need to check if the account has enough balance to execute our position order, then if able we then
    perform an action on the environment. The environment will return a position which me must then add to the
    account data.
    """
    while environment.step() is not None:
        # we first get the state of the environment
        state = environment.step()
        """
        Then now we need to act on this state so we parse it to our agent to make a decision on what to do...
        """
        action = agent.act(state)
        """
        Once we get an action we now need to check the account instance to see if there is enough money to do anything.
        """
        print(state)
        environment.__next__()
