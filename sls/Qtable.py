import numpy as np
import pandas as pd
from .agents.AbstractAgent import AbstractAgent
from sls import helper
import random


class Qtable():

    def __init__(self, actions, field_size=64, learning_rate=0.5, discount=0.75):
        self.actions = actions
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT = discount
        self.qtable = pd.DataFrame(0, index=helper.create_states(), columns=actions)

    def update_q_value(self, state, action, direction, reward, position, beacon_coord):
        # Q(s, a) <- Q(s, a) + a[r + Q(s, a)]
        current_q = self.qtable.loc[(helper.convert_array_to_string(state)), direction]
        future_state = position + action
        if future_state[0] < 0:
            future_state[0] = 0
        elif future_state[0] >= 64:
            future_state[0] = 63
        if future_state[1] < 0:
            future_state[1] = 0
        elif future_state[1] >= 64:
            future_state[1] = 63
        future = beacon_coord - (position + action)
        future[future > 0] = 1
        future[future < 0] = -1
        if reward > 0:
            new_q = current_q + self.LEARNING_RATE * (reward - current_q)
        else:
            best_future_state = max(self.qtable.loc[(helper.convert_array_to_string(future))])
            new_q = current_q + self.LEARNING_RATE * (reward + best_future_state * self.DISCOUNT - current_q)
        self.qtable.loc[helper.convert_array_to_string(state), direction] = new_q
        return self.qtable


    def get_best_action(self, state, direction):
        row =self.qtable.loc[helper.convert_array_to_string(state)]
        #If we have no single best option we choose a random one
        if(row.max() == 0):
            return random.choice(list(direction))
        return row.idxmax()
