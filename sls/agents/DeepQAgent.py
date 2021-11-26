
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
        self.states = helper.create_states()
        self.verbose = 0
        self.train = train
        self.action = 0
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
            if self._EPSILON - 1 / 500 > 0.1:
                self._EPSILON -= 1 / 500
            else:
                self._EPSILON = 0.1
            if ep > 500:
                self._EPSILON = 0
        return self._EPSILON

    def get_epsilon(self):
        return self._EPSILON

    def step(self, obs):
        self.step_count += 1
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            beacon_object = self._get_beacon(obs)
            beacon_coordinates = self._get_unit_pos(beacon_object)
            marine_coordinates = self._get_unit_pos(marine)
            distance = beacon_coordinates - marine_coordinates
            distance = np.array([(2 * (distance[0] - - 64) / (64 - -64) ) - 1, (2 * (distance[1] - - 64) / (64 - -64) ) - 1])
            p = random.random()

            if self.current_distance is not None:
                self.experience_replay.append(Experience(self.current_distance, self._LASTDIRECTION,
                                                         obs.reward,
                                                         obs.reward == 1,
                                                         distance))
            # Perform action
            if p < self._EPSILON and self.train:  # Choose random direction
                direction_key = np.random.choice(list(self._DIRECTIONS.keys()))

                # Update Q-table
            else:
                direction_key = self.network.model.predict(distance.reshape(-1, 2))[0]
                direction_key = list(self._DIRECTIONS.keys())[np.argmax(direction_key)]
            if self.experience_replay.__len__() > 6000 and self.train:
                exp_replay = self.experience_replay.sample(32)
                train_data = []
                for exp in exp_replay:
                    train_data.append(exp.state)
                labels = self.network.model.predict(np.array(np.array(train_data).reshape(-1, 2), dtype=np.float64))
                for i, exp in enumerate(exp_replay):
                    if exp.done:
                        labels[i][list(self._DIRECTIONS.keys()).index(exp.action)] = exp.reward
                    else:
                        labels[i][list(self._DIRECTIONS.keys()).index(exp.action)] = (exp.reward + self.learning_rate * np.max(self.target_network.model.predict(exp.new_state.reshape(-1, 2))[0]))
                self.network.model.fit(np.array(train_data, dtype=np.float64).reshape(-1, 2), np.array(labels, dtype=np.float64), verbose=self.verbose)
                self.verbose = 0

                if self.step_count > 200:
                    self.step_count = 0
                    self.target_network.model.set_weights(self.network.model.get_weights())
                    self.verbose = 1
                    print('reset')
            self.current_distance = distance
            self._LASTDIRECTION = direction_key
            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename='models'):
        self.network.save_model()
        pass

    def load_model(self, directory, filename='models'):
        self.network.load_model()
        pass
