import sys
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.expreplay import ExperienceReplay, PrioritizedReplayBuffer
from sls.expreplay import Experience
from sls.Qmodel import Model
import random
import tensorflow as tf
import pandas as pd
from decimal import *


class CNNAgent2(AbstractAgent):

    def __init__(self, screen_size, train=True):
        tf.compat.v1.disable_eager_execution()
        super(CNNAgent2, self).__init__(screen_size)
        self.save = './models/my_model_weights_final_cnn2.h5'
        self.actions = list(self._DIRECTIONS.keys())
        self.verbose = 0
        self.update_target_interval = 300
        self.min_exp_len = 6000
        self.batch_size = 32
        self.next_distance = None
        self.new_game = True
        self.train = train
        self.network = Model(2, train, is_conv_2=True)
        self.learning_rate = 0.9
        self.step_count = 0
        self.experience_replay_size = 100_000
        self.new_game = True
        self.current_state = None
        self._LASTDIRECTION = None
        self.alpha = 0.4
        self.beta = 0.6
        self.beta_inc = Decimal(5e-6)
        self.epsilon_replay = Decimal(1e-6)
        self.experience_replay = ExperienceReplay()
        if not self.train:
            self._EPSILON = 0
        else:
            self._EPSILON = 1
            self.target_network = Model(2, train, is_conv=True)

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
        marine = self._get_marine(obs)
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
            print(state)
            state = np.array(state)
            marine_coordinates = self._get_unit_pos(marine)
            if random.random() < self._EPSILON and self.train:  # Choose random direction
                direction_key = np.random.choice(self.actions)
                # Update Q-table
            else:
                q_values = self.network.model.predict(state.reshape([1, 16, 16, 1]))
                direction_key = self.actions[np.argmax(q_values)]

            if self.train and not self.new_game:
                self.experience_replay.append(Experience(self.current_distance, self._LASTDIRECTION,
                                                         obs.reward,
                                                         obs.last() or obs.reward == 1,
                                                         state))
            if self.experience_replay.__len__() > self.min_exp_len and self.train:
                self.train_agent()
            if obs.reward != 1 and not obs.last():
                self.current_state = state
                self._LASTDIRECTION = direction_key
                self.new_game = False
            else:
                self._LASTDIRECTION = self._NO_OP
                self.current_state = None
                self.new_game = True

            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def train_agent(self):
        exp_replay = self.experience_replay.sample(self.batch_size)
        states = np.array([exp.state for exp in exp_replay], dtype=np.float64)
        new_states = np.array([exp.new_state for exp in exp_replay],
                              dtype=np.float64)
        y = self.network.model.predict(states)
        y_new = self.target_network.model.predict(new_states)
        for i, exp in enumerate(exp_replay):
            if exp.done:
                y[i][self.actions.index(
                    exp.action)] = exp.reward
            else:
                y[i][self.actions.index(exp.action)] = (
                        exp.reward + self.learning_rate * max(y_new[i]))  # max(y_new[i])
        self.network.model.fit(states, y, verbose=self.verbose)

        self.verbose = 0

    def update_steps(self):
        if self.step_count % self.update_target_interval == 0 and self.train:
            self.target_network.model.set_weights(self.network.model.get_weights())
            self.verbose = 1
            print(self.beta)
        else:
            self.verbose = 0
            self.step_count += 1

    def save_model(self, filename='models'):
        self.network.model.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.network.model.load_weights(self.save)