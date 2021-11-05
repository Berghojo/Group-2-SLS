import os.path
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.Qtable import Qtable
import random
import datetime
import os


class QAgent(AbstractAgent):

    def __init__(self, screen_size, train=True, ):
        super(QAgent, self).__init__(screen_size)
        self.qtable = Qtable(self.get_directions().keys())
        self._EPSILON = 0.9

    def epsilon_decay(self, ep):
        if self._EPSILON - 1 / 1000 > 0.1:
            self._EPSILON -= 1 / 1000
        else:
            self._EPSILON = 0.1
        if ep > 1500:
            self._EPSILON = 0
        return self._EPSILON

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP

            beacon_object = self._get_beacon(obs)
            beacon_coordinates = self._get_unit_pos(beacon_object)
            marine_coordinates = self._get_unit_pos(marine)
            distance = beacon_coordinates - marine_coordinates
            p = random.random()
            # pull current best action
            direction_key = self.qtable.get_best_action(distance)
            # Perform action
            if p < self._EPSILON:
                # Choose random direction
                direction_key = np.random.choice(list(self._DIRECTIONS.keys()))
            # Update Q-table
            self.qtable.update_q_value(helper.search(self.qtable.DICTIONARY, distance), self._DIRECTIONS[direction_key],
                                       direction_key, obs.reward, marine_coordinates, beacon_coordinates)
            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.qtable.qtable.to_pickle("./pickles/qtable_" + datetime.datetime.now().strftime("%y%m%d") + ".pkl")
        pass

    def load_model(self, filename):
        pass
