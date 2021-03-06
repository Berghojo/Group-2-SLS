import os.path
import scipy
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.SarsaQtable import SarsaQtable
import random
import datetime
import os
import pandas as pd


class SarsaAgent(AbstractAgent):

    def __init__(self, screen_size, train=True):
        super(SarsaAgent, self).__init__(screen_size)
        self.qtable = SarsaQtable(self.get_directions().keys())
        self._EPSILON = 0.9
        self._TEMPERATURE = 0.5
        self._LASTDIRECTION = 'N'
        self.train = train
        self.action = 0
        self.current_distance = [0, 0]
        if not self.train:
            self._EPSILON = 0
        else:
            self._EPSILON = 1

    def epsilon_decay(self, ep):
        if self.train:
            if self._EPSILON - 1 / 500 > 0.1:
                self._EPSILON -= 1 / 500
            else:
                self._EPSILON = 0.1
            if ep > 500:
                self._EPSILON = 0
        return self._EPSILON

    def temperature_decay(self, ep):
        if self.train:
            if self._TEMPERATURE - 1 / 4000 > 0.1:
                self._TEMPERATURE -= 1 / 4000
            else:
                self._TEMPERATURE = 0.1
            if ep > 4000:
                self._TEMPERATURE = 0
        return self._TEMPERATURE

    def get_epsilon(self):
        return self._EPSILON

    def boltzmann_select(self, state):
        i = self.qtable.qtable.loc[state]
        q_dist = scipy.softmax(i / self._TEMPERATURE)
        action = np.random.choice(self._DIRECTIONS.keys(), p=q_dist)
        return action

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
            # Pull current best action

            # Perform action
            if p < self._EPSILON and self.train:  # Choose random direction
                direction_key = np.random.choice(list(self._DIRECTIONS.keys()))
                self.qtable.update_q_value(helper.search(self.qtable.DICTIONARY, self.current_distance),
                                           self._DIRECTIONS[self._LASTDIRECTION],
                                           self._LASTDIRECTION, obs.reward, self.current_distance)
            else:
                if self.train:  # Update Q-table
                    direction_key = self.qtable.get_best_action(distance)
                    self.qtable.update_q_value(helper.search(self.qtable.DICTIONARY,
                                                             self.current_distance),
                                               self._DIRECTIONS[self._LASTDIRECTION],
                                               self._LASTDIRECTION, obs.reward, self.current_distance)

                else:
                    direction_key = self.qtable.get_best_action(distance)
            # direction_key = self.boltzmann_select(helper.search(self.qtable.DICTIONARY, distance))

            if obs.reward == 1:
                print(self.current_distance)
                print('sad')
            self.current_distance = distance
            self._LASTDIRECTION = direction_key
            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        experiment_iteration = 'SARSA_5000'
        self.qtable.qtable.to_pickle("./pickles/qtable_" + datetime.datetime.now().strftime("%y%m%d_") +
                                     experiment_iteration + ".pkl")
        pass

    def load_model(self, directory, filename='qtable_211118_SARSA_5000.pkl'):
        qtable = pd.read_pickle(directory + filename)
        self.qtable.qtable = qtable
        pass
