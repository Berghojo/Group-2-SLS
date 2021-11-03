import numpy as np
import pandas as pd
from .agents.AbstractAgent import AbstractAgent

SPACE_SIZE = 64
LEARNING_RATE = 0.1
DISCOUNT = 1


def create_states(number):
    states = []
    for i in range(-number, number):
        for e in range(-number, number):
            states.append(" ".join(str(x) for x in [i, e]))
    return states


def convert_to_string(int_list):
    return " ".join(str(x) for x in int_list)

class Qtable():

    def __init__(self):
        #self.qtable = np.zeros((SPACE_SIZE*SPACE_SIZE, len(AbstractAgent.get_directions())))
        self.qtable = pd.DataFrame(0, index=create_states(64), columns=['S', 'N', 'W', 'E', 'SW', 'NE', 'SE', 'NW'])

    def update_q_value(self, state, action, direction):
        # Q(s, a) <- Q(s, a) + a[r + Q(s, a)]
        current_q = self.qtable.loc[(convert_to_string(state))][direction]
        if np.array_equal(state, [0, 0]):
            new_q = current_q + LEARNING_RATE * (1 - current_q)
        else:
            future_state = max(self.qtable.loc[(convert_to_string(state + action))])
            new_q = current_q + LEARNING_RATE * (-1 + future_state * DISCOUNT - current_q )

        self.qtable.loc[(convert_to_string(state))][direction] = new_q

        return new_q

    def get_best_action(self, state):
        return self.qtable.loc[convert_to_string(state)].idxmax()

