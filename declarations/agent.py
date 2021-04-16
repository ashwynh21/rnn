from random import random, randrange
from typing import List

from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import Accuracy
from keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

import numpy as np

"""
Here we are going to define and the problem that we are presented with so that we are able to understand the input data
that we are going to be interested in as our feature set. We are going to be exploring the possibility of using a deep
neural network with a few additions in the concept of Q-learning.
"""

"""
We begin by defining a class Agent to begin modelling our system, we are going to use this application as our training
ground for our forex model. So basically we need to define the class with the context of training in mind.
"""
scalar = MinMaxScaler()


class Agent(object):
    """
    We need to setup the way this thing is going to work when given training data. we will need to create an environment
    as close to a real trading account as possible.
    """

    def __init__(self, name: str):
        """
        Let us first define the hyper parameters for the agent that we will be using so here we need to use the
        bellman loss function to be able to evaluate the loss of our model properly, we also need to define our
        metrics that will be used to measure the performance of our system.

        This said we now need to consider how we are going to layout our metric system so that we are able to
        produce reports on the output data of the agent when the agent is performing trades.
        :param name:
        """
        # we init the account
        self.learning_rate = 0.001
        self.iteration = True
        self.name = name

        # affinity for long term reward
        self.gamma = 0.95
        self.actions = 3

        # our exploration hyper parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # we will use the adam optimizer since it uses the gradient descent algorithm to optimize the loss function.
        self.optimizer = Adam(lr=self.learning_rate)
        self.model = self.__model__()
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
        # self.model.add(LSTM(28, input_shape=(2, 5, 28)))
        model.add(Conv2D(14, (3, 3), padding='same', activation='sigmoid', input_shape=(12, 6, 1)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(28, (3, 3), activation='sigmoid'))
        model.add(MaxPooling2D((2, 1)))
        # we would like to add an LSTM layer here for it to remember the logical structure for future decisions that are
        # important
        model.add(Flatten())
        model.add(Reshape((1, -1)))

        model.add(LSTM(56))
        model.add(Dropout(0.5))

        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.actions))  # hold, buy, sell

        model.compile(loss=Huber(), optimizer=self.optimizer, metrics=[Accuracy()])

        return model

    def load(self):
        return load_model(f'models/{self.name}')

    def save(self):
        self.model.save(f'models/{self.name}.h5')

    def act(self, state: List[List[float]]):
        # let us first normalize the state before acting on it.
        state = self.normalize(state)

        if random() < self.epsilon or self.iteration:
            return randrange(self.actions)

        return np.argmax(self.model.predict(state))

    """
    we define a static method that will normalize input state so that it can be used by the model that we have defined
    """
    @staticmethod
    def normalize(data: List[List[float]]) -> np.array:
        # first we need to remove the first element of each element in the List
        cache = []
        time = []

        # we iterate through 12 elements and move each first element within the array
        for i in range(len(data)):
            time.append(data[i][0])
            cache.append(data[i][1:])

        # we will iterate through the data and normalize each
        scaled = scalar.fit_transform(cache).tolist()
        for i, candle in enumerate(scaled):
            candle.append(time[i])

        return np.array([np.array(scaled).reshape(12, 6, 1)])
