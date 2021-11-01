from sls.agents import AbstractAgent
import numpy as np


class RandomAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(RandomAgent, self).__init__(screen_size)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)

            d = np.random.choice(list(self._DIRECTIONS.keys()))
            return self._dir_to_sc2_action(d, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
