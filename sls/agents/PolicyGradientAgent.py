import sys
from pympler import asizeof
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.expreplay import ExperienceReplay
from sls.expreplay import Experience
from sls.Qmodel import Model
import random
from collections import namedtuple
import tensorflow as tf
import pandas as pd

SAR = namedtuple("SAR", ["state", "action", "reward"])


class PolicyAgent(AbstractAgent):
    def __init__(self, screen_size, train=True):
        tf.compat.v1.disable_eager_execution()
        super(PolicyAgent, self).__init__(screen_size)
        self.save = './models/policy_weights_final.h5'
        self.actions = list(self._DIRECTIONS.keys())
        self.verbose = 0
        self.update_target_interval = 300
        self.min_exp_len = 6000
        self.batch_size = 32
        self.reward_pos = 100
        self.new_episode = False
        self.reward_neg = -0.1
        self.next_distance = None
        self.new_game = True
        self.train = train
        self.network = Model(2, train)
        self.gamma = 0.99
        # self.gamma = 0.00025
        self.step_count = 0
        self.new_episode = False
        self.sar_batch = []
        self.experience_replay = ExperienceReplay()
        self.new_game = True
        self.current_distance = None
        self.current_action = None
        if not self.train:
            self._EPSILON = 0
        else:
            self._EPSILON = 1

    def epsilon_decay(self, ep):
        if self.train:
            if self._EPSILON - 1 / 500 > 0.05:
                self._EPSILON -= 1 / 500
            else:
                self._EPSILON = 0.05
        return self._EPSILON

    def get_epsilon(self):
        return self._EPSILON

    def step(self, obs):
        self.step_count += 1
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            reward = self.reward_pos if obs.reward > 0 else self.reward_neg
            if self.current_distance is not None:
                self.sar_batch.append(SAR(self.current_distance, self.current_action, reward))
            if obs.last() or reward > 0:
                reward_sum = []
                train_states = []
                actions = []
                for t, sar in enumerate(self.sar_batch):
                    reward = 0
                    for i in range(len(self.sar_batch) - t):
                        reward = reward + self.gamma ** i * self.sar_batch[i + t].reward

                    reward_sum.append([reward, self.actions.index(sar.action)])
                    train_states.append(sar.state)
                train_states = np.array(train_states)
                reward_sum = np.array(reward_sum)
                self.network.model.fit(train_states, reward_sum)
                self.new_episode = False
                self.current_distance = None
                self.sar_batch = []

            marine = self._get_marine(obs)
            beacon_object = self._get_beacon(obs)

            if beacon_object is None or marine is None:
                return self._NO_OP

            beacon_coordinates = self._get_unit_pos(beacon_object)
            marine_coordinates = self._get_unit_pos(marine)
            self.current_distance = np.array((beacon_coordinates - marine_coordinates) / self.screen_size)
            policy_probability = self.network.model.predict(np.array([self.current_distance]))
            policy_probability = np.squeeze(policy_probability)
            direction_key = np.random.choice(self.actions, p=policy_probability)
            self.current_action = direction_key
            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename='models'):
        self.network.model.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.network.model.load_weights(self.save)
