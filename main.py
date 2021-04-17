from declarations import Account
from declarations.config import config
from declarations.agent import Agent
from declarations.environment import Environment
from tqdm import tqdm

from declarations.experience import Experience
from declarations.metric import Metric

if __name__ == '__main__':
    # init an agent, environment, and the account that the agent will be trading on the environment.
    agent = Agent('nasdaq', 128)
    environment = Environment('assets/NAS100.csv', 'assets/NAS100_2021.csv')
    account = Account(100)
    metric = Metric()

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
        index, state = environment.step()
        """
        Then now we need to act on this state so we parse it to our agent to make a decision on what to do...
        """
        action = agent.act(state)
        """
        Once we get an action we now need to check the account instance to see if there is enough money to do anything.
        """
        if account.isable(state.price(), 0.01):
            """
            Once confirmed then we place our oder onto the environment.
            """
            if action.action == 0:
                # then we get the position from the environment and add it to the account
                account.record(str(index), environment.buy(0.01, account.balance))
            elif action.action == 1:
                account.record(str(index), environment.sell(0.01, account.balance))
            elif action.action == 2:
                # here we need to get the state result of the action
                environment.__next__()

                # for holding we will give the agent a reward
                memory = Experience(state=state, action=action, reward=0.5, next=environment.step())
                # then we add this experience to the agents memory.
                agent.memory.remember(memory)

            """
            Once the states market position has been handled then we need to now check our closing strategy for each
            of our open positions.
            """
            for k, p in account.closable().items():
                r = environment.close(p)
                # now we update the account by removing the positions and adding the result to the ledger
                account.archive(k, r)

                # now we need to remember the position in the memory so we construct an experience.
                experience = Experience(p.state, p.action, r.reward(), environment.step())
                agent.memory.remember(experience)

            # so if the memory is full we should start training the model
            if agent.memory.isready() and index % 128 == 0:
                # get the loss and add it to the metrics we are going to observe
                metric.addloss(agent.experience())
        else:
            """
            If the account runs out of money we end the session and the agent must start afresh.
            
            Now in this case we need to consider the metrics of evaluating the performance of our algorithm
            that is, we need to create a class that will run this operation for us. 
            """
            print(f'Not Enough Money Resetting Account - {index}')
            account.reset(100)
            environment.reset()

            # update our metrics
            metric.restart()
            print(metric.survival(index))
