from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.expreplay import ExperienceReplay
from sls.expreplay import Experience
from sls.Qmodel import Model
import random
import tensorflow as tf
import pandas as pd


class DeepQAgent(AbstractAgent):

    def __init__(self, screen_size, train=True):
        tf.compat.v1.disable_eager_execution()
        tf.autograph.set_verbosity(0)
        super(DeepQAgent, self).__init__(screen_size)
        self._LASTDIRECTION = None
        self.actions = list(self._DIRECTIONS.keys())
        self.states = np.array(helper.create_states())
        self.verbose = 0
        self.update_target_interval = 300
        self.min_exp_len = 6000
        self.batch_size = 1024
        self.next_distance = None
        self.new_game = True
        self.train = train
        self.action = None
        self.current_distance = None
        self.experience_replay = ExperienceReplay()
        self.network = Model(2, train)
        self.target_network = Model(2, train)
        self.learning_rate = 0.5
        self.step_count = 0
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
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            beacon_object = self._get_beacon(obs)
            beacon_coordinates = self._get_unit_pos(beacon_object)
            marine_coordinates = self._get_unit_pos(marine)
            distance = np.array(beacon_coordinates - marine_coordinates)/64
            if self.current_distance is not None and self.train and not self.new_game:
                self.experience_replay.append(Experience(self.current_distance, self._LASTDIRECTION,
                                                         obs.reward,
                                                         obs.last or obs.reward == 1,
                                                         distance))
            # Perform action
            if random.random() < self._EPSILON and self.train:  # Choose random direction
                direction_key = np.random.choice(self.actions)
                # Update Q-table
            else:
                q_values = self.network.model.predict(distance.reshape(1, 2))
                direction_key = self.actions[np.argmax(q_values)]

            if self.experience_replay.__len__() > self.min_exp_len and self.train:
                self.train_agent()
            if obs.reward != 1 and not obs.last():
                self.current_distance = distance
                self._LASTDIRECTION = direction_key
                self.new_game = False
            else:
                self._LASTDIRECTION = 'NO_OP'
                self.current_distance = None
                self.new_game = True

            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def train_agent(self):
        exp_replay = self.experience_replay.sample(self.batch_size)
        states = [exp.state for exp in exp_replay]
        new_states = [exp.new_state for exp in exp_replay]
        states = np.array(states, dtype=np.float64).reshape(self.batch_size, 2)
        new_states = np.array(new_states,
                              dtype=np.float64) .reshape(self.batch_size, 2)# TODO transformation in external function
        y = self.network.model.predict(states)
        y_new = self.target_network.model.predict(new_states)
        for i, exp in enumerate(exp_replay):
            if exp.done:
                y[i][self.actions.index(
                    exp.action)] = exp.reward  # TODO save action as numeric value instead of char
            else:
                y[i][self.actions.index(exp.action)] = (
                        exp.reward + self.learning_rate * max(y_new[i]))
        self.network.model.fit(states, np.array(y), verbose=self.verbose)
        self.verbose = 0

    def update_steps(self):
        if self.step_count > self.update_target_interval:
            self.target_network.model.set_weights(self.network.model.get_weights())
            self.verbose = 1
            self.step_count = 0
        else:
            self.step_count += 1
    def save_model(self, filename='models'):
        self.network.model.save_weights('./models/my_model_weights3.h5')

    def load_model(self, directory, filename='models'):
        self.network.model.load_weights('./models/my_model_weights3.h5')
