from random import random, randrange

from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import Accuracy
from keras.losses import Huber
from tensorflow.python.keras.models import load_model

import numpy as np

from declarations.action import Action
from declarations.memory import Memory
from declarations.state import State

"""
Here we are going to define and the problem that we are presented with so that we are able to understand the input data
that we are going to be interested in as our feature set. We are going to be exploring the possibility of using a deep
neural network with a few additions in the concept of Q-learning.
"""

"""
We begin by defining a class Agent to begin modelling our system, we are going to use this application as our training
ground for our forex model. So basically we need to define the class with the context of training in mind.
"""


class Agent(object):
    # so we define a memory property for the agent to add its own experiences
    memory: Memory
    """
    We need to setup the way this thing is going to work when given training data. we will need to create an environment
    as close to a real trading account as possible.
    """

    def __init__(self, name: str, batch: int, evaluate=False):
        """
        Let us first define the hyper parameters for the agent that we will be using so here we need to use the
        bellman loss function to be able to evaluate the loss of our model properly, we also need to define our
        metrics that will be used to measure the performance of our system.

        This said we now need to consider how we are going to layout our metric system so that we are able to
        produce reports on the output data of the agent when the agent is performing trades.
        :param name:
        """
        self.memory = Memory(batch)
        # we define a parameter to update the target weights.
        self.update = 10
        self.sessions = 0

        # we init the account
        self.learning_rate = 0.0005
        self.name = name

        # affinity for long term reward
        self.gamma = 0.95
        self.actions = 3

        # we define evaluate in case we are evaluating the Agent
        self.evaluate = evaluate

        # our exploration hyper parameters
        self.epsilon = 1.0
        # since there are few choices to make we will want to have a 5% decay
        self.epsilon_decay = 0.99

        # we will use the adam optimizer since it uses the gradient descent algorithm to optimize the loss function.
        self.optimizer = Adam(lr=self.learning_rate)
        # we then define our policy model as well as our target model.
        self.policy = self.__model__()
        self.target = self.__model__()
        # then we make sure the weights are updated on the target model.
        self.__update__()

        """
        Since we are using Q-Learning and we will be using t-dqn strategy for learning we will initialize a few values
        forward
        """

    """
    with the model functionality done we now need to work on how this model is going to learn from a said data set X
    """

    def __model__(self):
        """
        We define a function that will return our model, this is a hidden function that should only be used in the
        constructor of the agent system.
        :return:
        """
        # we also need to define our model.
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', activation='sigmoid', input_shape=(120, 6, 1)))
        model.add(MaxPooling2D((2, 1)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='sigmoid'))
        model.add(MaxPooling2D((2, 1)))

        model.add(Conv2D(96, (3, 3), padding='same', activation='sigmoid'))
        model.add(MaxPooling2D((5, 1)))

        # increasing complexity to find and better solution to the decision problem.
        model.add(Conv2D(64, (3, 3), padding='same', activation='sigmoid'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same', activation='sigmoid'))
        model.add(MaxPooling2D((3, 3)))
        # we would like to add an LSTM layer here for it to remember the logical structure for future decisions that are
        # important

        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))

        model.add(Dense(units=self.actions))  # hold, buy, sell

        model.compile(loss=Huber(), optimizer=self.optimizer, metrics=[Accuracy()])

        return model

    """
    We define a function that will update the target model weights
    """

    def __update__(self):
        self.target.set_weights(self.policy.get_weights())

    def load(self):
        self.policy = load_model(f'models/{self.name}.h5')
        self.__update__()
        return self.policy

    def save(self):
        self.policy.save(f'models/{self.name}.h5')

    def act(self, state: State) -> Action:
        # let us first normalize the state before acting on it.
        if random() < self.epsilon and not self.evaluate:
            p = list(np.zeros(self.actions, dtype=float).flatten())
            p[randrange(self.actions)] = 1

            return Action([[[p]]])
        return Action(self.policy.predict(state.normalize()), False)

    """
    We now need to define a function that will learn from the saved experiences in memory.
    """

    def experience(self) -> float:
        buffer = self.memory.recall(self.memory.maxsize())

        x, y = [], []
        for state, action, reward, nexter in buffer:
            # estimate q-values based on current state

            target = Action(self.policy.predict(state.normalize()))
            bullseye = Action(self.target.predict(nexter.normalize()))

            # update the target for current action based on discounted reward
            target.probabilities[0][0][0][action.action] = reward + self.gamma * np.amax(bullseye.probabilities)
            x.append(state.normalize()[0])
            y.append(target.normalize()[0])

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        self.epsilon *= self.epsilon_decay

        # check to see if its time to update the target model weights.
        if self.sessions % self.update == 0:
            self.__update__()
            # we should also use the opportunity to save the policy model.
            self.save()

        self.sessions = self.sessions + 1

        # update q-function parameters based on huber loss gradient
        loss = self.policy.fit(
            np.array(x), np.array(y),
            epochs=1, verbose=0
        ).history["loss"][0]

        return loss
