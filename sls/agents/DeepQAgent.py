import os.path
import scipy
from sls import helper
from tensorflow.keras import callbacks
from sls.agents import AbstractAgent
import numpy as np
from sls.expreplay import ExperienceReplay
from sls.expreplay import Experience
from sls.Qmodel import Model
import random
import datetime
import os
import pandas as pd


class DeepQAgent(AbstractAgent):

    def __init__(self, screen_size, train=True):
        super(DeepQAgent, self).__init__(screen_size)
        self._LASTDIRECTION = None
        self.states, self.indexes = helper.create_states()
        self.train = train
        self.action = 0
        self.current_distance = None
        self.experience_replay = ExperienceReplay()
        self.network = Model(1)
        self.target_network = Model(1)
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
            p = random.random()

            if self.current_distance is not None:
                self.experience_replay.append(Experience(helper.search(self.states, self.current_distance),self._LASTDIRECTION,
                                          obs.reward,
                                          obs.reward == 1,
                                          helper.search(self.states, distance)))
            # Perform action
            if p < self._EPSILON and self.train:  # Choose random direction
                direction_key = np.random.choice(list(self._DIRECTIONS.keys()))

                # Update Q-table
            else:
                direction_key = self.network.model.predict([helper.search(self.states, distance)])[0]
                direction_key = list(self._DIRECTIONS.keys())[np.argmax(direction_key)]
            if self.experience_replay.__len__() > 32 and self.train:
                if self.experience_replay.__len__() > 32:
                    print('LEARNING')

                exp_replay = self.experience_replay.sample(32)
                labels = []
                train_data = []
                for exp in exp_replay:
                    train_data.append(exp.state)
                    if exp.done:
                        labels.append(exp.reward)
                    else:
                        labels.append(exp.reward + self.learning_rate * np.max(self.target_network.model.predict([exp.new_state])[0]))

                #EPOCHS = 10
                #checkpoint_path = 'models/'
                #callback = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True,
                #                                      save_freq='epoch', verbose=1)
                #self.network.model.summary()
                self.network.model.fit(train_data, labels)
                if self.step_count > 300:
                    self.step_count = 0
                    self.target_network = self.network

            self.current_distance = distance
            self._LASTDIRECTION = direction_key
            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename='test'):
        self.network.save_model()
        pass

    def load_model(self, directory, filename='qtable_211118_SARSA_5000.pkl'):
        qtable = pd.read_pickle(directory + filename)
        self.qtable.qtable = qtable
        pass
