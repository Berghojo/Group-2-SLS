import os.path
import datetime
from sls.agents import AbstractAgent
import numpy as np
from sls.Qtable import Qtable
import random
import pandas as pd


class QAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QAgent, self).__init__(screen_size)
        self.qtable = Qtable(self.get_directions().keys())
        self.train = train
        if not self.train:
            print("Were not training")

            self._EPSILON = 0
        else:
            self._EPSILON = 1

    def epsilon_decay(self, ep):
        if self._EPSILON - 1 / 3000 > 0.1:
            self._EPSILON -= 1 / 3000
        else:
            self._EPSILON = 0.1
        if ep > 4000:
            self._EPSILON = 0
        return self._EPSILON

    def get_epsilon(self):
        return self._EPSILON

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            beacon_object = self._get_beacon(obs)
            beacon_coords = self._get_unit_pos(beacon_object)
            marine_coords = self._get_unit_pos(marine)
            distance = beacon_coords - marine_coords
            distance[distance > 0] = 1
            distance[distance < 0] = -1
            p = random.random()
            direction_key = self.qtable.get_best_action(distance)
            # Choose random direction
            if p < self._EPSILON:
                rest_direction_keys = list(self._DIRECTIONS.keys())
                rest_direction_keys.remove(direction_key)
                direction_key = np.random.choice(rest_direction_keys)

            # Update Q-table
            if self.train:
                self.qtable.update_q_value(distance, self._DIRECTIONS[direction_key], direction_key, obs.reward,
                                           marine_coords, beacon_coords)

            # Perform action
            return self._dir_to_sc2_action(direction_key, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        experiment_iteration = '1'
        self.qtable.qtable.to_pickle("./pickles/qtable_" + datetime.datetime.now().strftime("%y%m%d_") +
                                     experiment_iteration + ".pkl")
        pass

    def load_model(self, directory, filename='qtable.pkl'):
        qtable = pd.read_pickle(directory + filename)
        self.qtable.qtable = qtable
        pass
