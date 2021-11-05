from abc import abstractmethod

import numpy as np
from pysc2.lib import actions, features


class AbstractAgent:
    _DIRECTIONS = {'N': [0, -1],
                   'NE': [1, -1],
                   'E': [1, 0],
                   'SE': [1, 1],
                   'S': [0, 1],
                   'SW': [-1, 1],
                   'W': [-1, 0],
                   'NW': [-1, -1]}

    """Sc2 Actions"""
    _MOVE_SCREEN = actions.FUNCTIONS.Move_screen
    _NO_OP = actions.FUNCTIONS.no_op()
    _SELECT_ARMY = actions.FUNCTIONS.select_army("select")



    def get_directions(self):
        return self._DIRECTIONS

    def __init__(self, screen_size):
        self.screen_size = screen_size

    @abstractmethod
    def step(self, obs):
        ...

    @abstractmethod
    def save_model(self, path):
        ...
    @abstractmethod
    def epsilon_decay(self, ep):
        ...
    @abstractmethod
    def load_model(self, path):
        ...

    def _get_beacon(self, obs):
        """Returns the unit obj representation of the beacon"""
        beacon = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.NEUTRAL)
        return beacon

    def _get_marine(self, obs):
        """Returns the unit obj representation of the marine"""
        marine = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.SELF)
        return marine

    def _get_unit_pos(self, unit):
        """Returns the (x, y) position of a unit obj"""
        return np.array([unit.x, unit.y])

    def _dir_to_sc2_action(self, d, marine_center):
        """Takes the direction the marine should walk and outputs an action for PySC2"""

        def _xy_offset(start, offset_x, offset_y):
            """Return point (x', y') offset from start.
               Pays attention to not set the point off beyond the screen border"""
            dest = start + np.array([offset_x, offset_y])
            if dest[0] < 0:
                dest[0] = 0
            elif dest[0] >= self.screen_size:
                dest[0] = self.screen_size - 1
            if dest[1] < 0:
                dest[1] = 0
            elif dest[1] >= self.screen_size:
                dest[1] = self.screen_size - 1
            return dest

        if d in self._DIRECTIONS.keys():
            next_pos = _xy_offset(marine_center,
                                  self.screen_size * self._DIRECTIONS[d][0],
                                  self.screen_size * self._DIRECTIONS[d][1])
            return self._MOVE_SCREEN("now", next_pos)
        else:
            return self._NO_OP
