import numpy as np
import pandas as pd
from .agents.AbstractAgent import AbstractAgent
from sls import helper
import random


class SarsaQtable():

    def __init__(self, actions, learning_rate=0.5, discount=0.75):
        self.actions = actions
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT = discount
        states, indexes = helper.create_states()
        self.qtable = pd.DataFrame(0, index=indexes, columns=actions)
        self.DICTIONARY = states

    def update_q_value(self, state, action, direction_key, reward, position, beacon_coordinates):
        # Q(s, a) <- Q(s, a) + a[r + Q(s, a)]
        current_q = self.qtable.loc[state, direction_key]
        future_position = position + action
        future_position[future_position < 0] = 0
        future_position[future_position > 63] = 63
        future_state_action = self.get_best_action(future_position)
        future_state = helper.search(self.DICTIONARY, future_position)
        future_q = self.qtable.loc[future_state, future_state_action]
        if reward > 0:
            new_q = current_q + self.LEARNING_RATE * (reward - current_q)
        else:
            new_q = current_q + self.LEARNING_RATE * (reward + future_q * self.DISCOUNT - current_q)
        self.qtable.loc[state, direction_key] = new_q
        return future_state_action

    def get_best_action(self, direction_key):
        direction_key = list(direction_key)
        row = self.qtable.loc[helper.search(self.DICTIONARY, direction_key)]
        # If we have no single best option we choose a random one
        row = row.sample(frac=1)
        return row.idxmax()
