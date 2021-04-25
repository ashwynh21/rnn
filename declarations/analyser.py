import pandas as pd
import matplotlib.pyplot as pt
from scipy.interpolate import interp1d


class Analyser(object):
    def __init__(self, data: str):
        # we define the initial time step state to zero
        self.__step = 0
        """
        So here we will define the input data set, from the a file path to the CSV learning and testing data.
        """
        self.__data = pd.read_csv(data)

    def __close__(self, field: str):
        """
        Here we are going interpolate the input data set that we have into 12 times more data and index the data
        accordingly.
        :return:
        """
        time = [x.timestamp() for x in list(pd.to_datetime(self.__data['time']))]

        price = list(self.__data[field])
        return time, price

    def plot(self, field: str):

        def timer(time):
            for i in range(len(time) - 1):
                a, b = int(time[i]), int(time[i + 1])
                yield from range(a, b, int((b - a) / 12))

        data = self.__close__(field)
        time = list(timer(data[0]))

        f = interp1d(data[0], data[1], kind='cubic')

        pt.plot(time, f(time), '--', )
        pt.show()
