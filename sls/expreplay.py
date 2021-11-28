import numpy as np
import collections
import sys

Experience = collections.namedtuple('Experience',
field_names = ['state', 'action', 'reward',
'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        if self.__len__() > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        return [self.buffer[idx] for idx in indices]