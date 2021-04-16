
"""
We define this class to store the (state, action reward, next) of a particular move
"""
from collections import deque
from typing import List
from random import sample

from declarations.experience import Experience


class Memory:
    __memory: deque

    def __init__(self):
        self.__memory = deque(maxlen=512)

    def remember(self, experience: Experience):
        # first we check if the memory isnt full.
        if len(self.__memory) == 512:
            # the we pop the end and push to the front.
            self.__memory.pop(0)

        return self.__memory.append(experience)

    def recall(self, size: int) -> List[Experience]:
        # this function is to sample the memory of the deque object.
        return sample(self.__memory, size)
