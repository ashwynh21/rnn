"""
In this file we will define a class that will allow us to predict the price from given data.

With this functionality we will use the prediction along withe price level that we have, together with
the agent to decide and confirm
"""
import math

from typing import List
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.metrics import Accuracy
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.losses import MeanSquaredError
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

scalar = MinMaxScaler()


class Price(object):

    def __init__(self, name: str):
        self.name = name
        self.learning_rate = 0.0005

        self.optimizer = Adam(lr=self.learning_rate)
        self.model = self.__model__()

    def __model__(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dense(1))

        model.compile(loss=MeanSquaredError(), optimizer=self.optimizer, metrics=[Accuracy()])

        return model

    def __update__(self):
        self.model.set_weights(self.model.get_weights())

    def load(self):
        self.model = load_model(f'models/{self.name}.h5')
        self.__update__()
        return self.model

    def save(self):
        self.model.save(f'models/{self.name}.h5')

    def train(self, data: List[List[float]]):
        train_len = math.ceil(len(data) * 0.8)
        training_data = scalar.fit_transform(data[:train_len])

        x, y = [], []
        for i in range(60, len(training_data)):
            x.append(training_data[i - 60:i, 0])
            y.append(training_data[i, 0])

        x, y = np.array(x), np.array(y)
        x = x.reshape((x.shape[0], x.shape[1], 1))

        self.model.fit(x, y, epochs=48, batch_size=32)

    def test(self, data: List[List[float]]):
        train_len = math.ceil(len(data) * 0.2)

        test_data = scalar.fit_transform(data[train_len - 60:, :])

        x = []
        for i in range(60, len(test_data)):
            x.append(test_data[i - 60:i, 0])

        x = np.array(x)
        x = x.reshape((x.shape[0], x.shape[1], 1))

        predictions = self.model.predict(x)
        predictions = scalar.inverse_transform(predictions)

        return np.array(predictions).flatten()

    @staticmethod
    def plot(data: List[float], predictions: List[float]):
        plt.plot(range(len(predictions)), data[-len(predictions):], label='Original')
        plt.plot(range(len(predictions)), predictions, label='Prediction')
        plt.legend()
        plt.show()

