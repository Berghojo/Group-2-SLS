import numpy as np
import collections
Experience = collections.namedtuple('Experience',
field_names = ['state', 'action', 'reward',
'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity=100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)

        return [self.buffer[idx] for idx in indices]