import numpy as np
import pandas as pd
from .agents.AbstractAgent import AbstractAgent
from sls import helper


class Qtable():

    def __init__(self, actions, field_size=64, learning_rate=1, discount=0.9):
        self.actions = actions
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT = discount
        self.qtable = pd.DataFrame(0, index=helper.create_states(field_size), columns=actions)

    def update_q_value(self, state, action, direction, reward):
        # Q(s, a) <- Q(s, a) + a[r + Q(s, a)]
        current_q = self.qtable.loc[(helper.convert_array_to_string(state))][direction]
        if reward > 0:
            new_q = current_q + self.LEARNING_RATE * (reward - current_q)
        else:
            future_state = max(self.qtable.loc[(helper.convert_array_to_string(state + action))])
            new_q = current_q + self.LEARNING_RATE * (reward + future_state * self.DISCOUNT - current_q)
        self.qtable.loc[helper.convert_array_to_string(state), direction] = new_q
        return self.qtable

    def get_best_action(self, state):
        return self.qtable.loc[helper.convert_array_to_string(state)].idxmax()
