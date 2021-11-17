from sls.agents import AbstractAgent
import numpy as np


class BasicAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            beacon_object = self._get_beacon(obs)
            beacon_coords = self._get_unit_pos(beacon_object)
            direction = beacon_coords - marine_coords
            direction[direction > 0] = 1
            direction[direction < 0] = -1

            # Qtable array rows = direction

            for key in self._DIRECTIONS:
                movement = self._DIRECTIONS[key]
                if np.array_equal(movement, direction):
                    d = key

            return self._dir_to_sc2_action(d, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
