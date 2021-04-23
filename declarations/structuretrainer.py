"""
We define the following class to isolate the trainer code of different agents that we will be building. We are moving
into building cordinated agents.
"""

from declarations import Environment, Agent, Account, Metric, Experience


class StructureTrainer:
    def train(self):
        # init an agent, environment, and the account that the agent will be trading on the environment.
        account = Account(100)
        metric = Metric()

        environment = Environment('assets/NAS100.csv')
        agent = Agent('nasdaq', 128)
        # run our training function
        while metric.restarts != 0:
            self.__train__(environment, agent, account, metric)

            ax = metric.actionsummary()

            print(f'+----------------------------------------------------------+')
            print(f'| Training Session Summary on Agent                        |')
            print(f'+----------------------------------------------------------+')
            print(f'| Restarts        | {str(metric.restarts).ljust(39)}|')
            print(f'+----------------------------------------------------------+')
            print(f'| Actions Summary | {str(len(metric.actions)).ljust(39)}|')
            print(f'+----------------------------------------------------------+')
            print(
                f'| Buy |    {str(ax["buy"]).ljust(4)}    | Sell |    {str(ax["sell"]).ljust(4)}    | Hold |    {str(ax["hold"]).ljust(4)}    |')
            print(f'+----------------------------------------------------------+\n')

            environment.reset()
            metric.reset()

    @staticmethod
    def __train__(environment: Environment, agent: Agent, account: Account, metric: Metric):
        """
        Now we need to devise a way that the agent will trade the account, by getting the state from the environment and
        acting on it using the account. Having made a choice on the action to take then we need to use the result action
        with the account.

        So we then need to check if the account has enough balance to execute our position order, then if able we then
        perform an action on the environment. The environment will return a position which me must then add to the
        account data.
        """
        while not environment.done():
            # we first get the state of the environment
            index, state = environment.step()
            """
            Then now we need to act on this state so we parse it to our agent to make a decision on what to do...
            """
            action = agent.act(state)

            metric.countaction(action)
            # then we need to check if the account is able to open positions otherwise the account is blown
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
                    memory = Experience(state=state, action=action, reward=-0.2, next=environment.step()[1])
                    # then we add this experience to the agents memory.
                    agent.memory.remember(memory)

                """
                Once the states market position has been handled then we need to now check our closing strategy for each
                of our open positions.
                """
                for k, p in account.closable(state, action).items():
                    r = environment.close(p)
                    # now we update the account by removing the positions and adding the result to the ledger
                    account.archive(k, r)

                    # now we need to remember the position in the memory so we construct an experience.
                    experience = Experience(p.state, p.action, r.reward(), environment.step()[1])
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
                # update our metrics
                metric.restart()
                # metric.reset()

                account.reset(100)
                # environment.reset()

                metric.survival(index)
                # we wont reset this time...

        metric.addprofit(account.balance - 100)
        # print(f'Session Complete - {account.balance}')

    """
    the function below is designed to evaluate the survival ability learned from the training session using a different
    data set from the next year.
    """
    @staticmethod
    def ___evaluate__(environment: Environment, agent: Agent, account: Account, metric: Metric):
        # first we need to setup the test environment so that we are able to evaluate our model properly.
        while not environment.done():
            # we first get the state of the environment
            index, state = environment.step()
            """
            Then now we need to act on this state so we parse it to our agent to make a decision on what to do...
            """
            action = agent.act(state)
            print(f'looping - {action.action}')
            """
            Once we get an action we now need to check the account instance to see if there is enough money to do
            anything.
            """
            # then we need to check if the account is able to open positions otherwise the account is blown
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

                """
                Once the states market position has been handled then we need to now check our closing strategy for each
                of our open positions.
                """
                for k, p in account.closable(state, action).items():
                    r = environment.close(p)
                    # now we update the account by removing the positions and adding the result to the ledger
                    account.archive(k, r)
            else:
                # update our metrics
                metric.restart()
                ma = metric.survival(index)

                print(f'Test Failed - Survived {index} Hours, MA - {ma[-1] if len(ma) > 0 else 0}')

        """
        So if it manages to get to here then it eventually learned to trade through the year given the initial deposit
        on the account.
        """
        metric.addprofit(account.balance - 100)
        print(f'Test Complete - {account.balance}')
