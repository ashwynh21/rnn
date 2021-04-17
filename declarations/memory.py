
"""
We define this class to store the (state, action reward, next) of a particular move
"""
from collections import deque
from typing import List
from random import sample

from declarations.experience import Experience


class Memory:
    __memory: deque
    __size: int

    def __init__(self, batch: int):
        self.__batch = batch
        self.__size = batch * 2
        self.__memory = deque(maxlen=self.__size)

    def remember(self, experience: Experience):
        # first we check if the memory isnt full.
        if len(self.__memory) == self.__size - 1:
            # the we pop the end and push to the front.
            for s in sample(self.__memory, self.__batch):
                self.__memory.remove(s)

        return self.__memory.append(experience)

    def recall(self, size: int) -> List[Experience]:
        # this function is to sample the memory of the deque object.
        return sample(self.__memory, size)

    def size(self) -> int:
        return len(self.__memory)

    def maxsize(self) -> int:
        return self.__batch

    def isready(self) -> bool:
        return len(self.__memory) >= self.__batch
