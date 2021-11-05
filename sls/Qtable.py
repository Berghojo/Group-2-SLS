import numpy as np
import pandas as pd
from .agents.AbstractAgent import AbstractAgent
from sls import helper
import random


class Qtable():

    def __init__(self, actions, field_size=64, learning_rate=0.1, discount=0.75):
        self.actions = actions
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT = discount
        states,indexes = helper.create_states()
        self.qtable = pd.DataFrame(0, index=indexes, columns=actions)
        self.DICTIONARY = states

    def update_q_value(self, state, action, direction, reward, position, beacon_coord):
        # Q(s, a) <- Q(s, a) + a[r + Q(s, a)]

        current_q = self.qtable.loc[state, direction]

        future_state = helper.search(self.DICTIONARY, beacon_coord - position + action)
        future = future_state
        if reward > 0:
            new_q = current_q + self.LEARNING_RATE * (reward - current_q)
        else:
            best_future_state = max(self.qtable.loc[future])
            new_q = current_q + self.LEARNING_RATE * (reward + best_future_state * self.DISCOUNT - current_q)
        self.qtable.loc[state, direction] = new_q
        return self.qtable


    def get_best_action(self, state, direction, navigation):

        row =self.qtable.loc[helper.search(self.DICTIONARY, direction)]

        #If we have no single best option we choose a random one
        if(row.max() == 0):
            return random.choice(list(navigation))
        return row.idxmax()
