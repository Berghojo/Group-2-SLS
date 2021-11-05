from sls.agents import AbstractAgent
import numpy as np
from sls.Qtable import Qtable
import random

EPSILON = 0.9


class QAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QAgent, self).__init__(screen_size)
        self.qtable = Qtable(self.get_directions().keys())

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            beacon_object = self._get_beacon(obs)
            beacon_coords = self._get_unit_pos(beacon_object)
            distance = beacon_coords - marine_coords

            # Choose random direction
            p = random.random()

            # Perform action
            if p < EPSILON:
                d = np.random.choice(list(self._DIRECTIONS.keys()))
            else:  # pull current best action
                d = self.qtable.get_best_action(distance)

            # Measure reward (step -1 & beacon +1)
            self.qtable.update_q_value(distance, self._DIRECTIONS[d], d, obs.reward)

            # Update Q-table

            return self._dir_to_sc2_action(d, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
