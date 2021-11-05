import os.path

from sls.agents import AbstractAgent
import numpy as np
from sls.Qtable import Qtable
import random



class QAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QAgent, self).__init__(screen_size)
        self.qtable = Qtable(self.get_directions().keys())
        self._EPSILON = 1

    def epsilon_decay(self, ep):
        if self._EPSILON-1/50 > 0.1:
            self._EPSILON -= 1/50
        else:
            self._EPSILON = 0.1
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
            # Choose random direction
            p = random.random()

            # Perform action
            if p < self._EPSILON:
                d = np.random.choice(list(self._DIRECTIONS.keys()))
            else:  # pull current best action
                d = self.qtable.get_best_action(distance, self._DIRECTIONS.keys())

            # Measure reward (step -1 & beacon +1)
            self.qtable.update_q_value(distance, self._DIRECTIONS[d], d, obs.reward, marine_coords, beacon_coords)
            #print(self.qtable.qtable)
            # Update Q-table

            return self._dir_to_sc2_action(d, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.qtable.qtable.to_pickle("./pickles/test.pkl")
        pass

    def load_model(self, filename):
        pass
