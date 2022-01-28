import sys
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.A2Cmodel import A2CModel
import tensorflow as tf
from collections import namedtuple

SAR = namedtuple("SAR", ["state", "action", "reward"])


class A2CAgent(AbstractAgent):
    def __init__(self, screen_size, train=True):
        tf.compat.v1.disable_eager_execution()
        super(A2CAgent, self).__init__(screen_size)
        self.save = './models/A2C_weights_final.h5'
        self.actions = list(self._DIRECTIONS.keys())
        self.train = train
        self.network = A2CModel(2, train)
        self.mini_batch_size = 64
        self.learning_rate = 0.0007
        self.entropy_constant = 0.005
        self.value_constant = 0.5
        self.n_step_return = 5
        self.sar_batch = []
        self.current_state = None
        self.new_episode = True
        self.entropy_loss = 0

    def step(self, obs):
        self.step_count += 1
        if self._MOVE_SCREEN.id in obs.observation.available_actions:

            marine = self._get_marine(obs)
            beacon_object = self._get_beacon(obs)

            if beacon_object is None or marine is None:
                return self._NO_OP

            marine_coordinates = self._get_unit_pos(marine)

            self.current_state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
            policy_probability, value = self.network.model.predict([np.array(self.current_state)])
            policy_probability = np.squeeze(policy_probability)
            direction_key = np.random.choice(self.actions, p=policy_probability)

            entropy = -np.sum(policy_probability * np.log(policy_probability))
            self.entropy_loss += entropy

            if self.train:
                self.train_agent(obs, value, policy_probability[self.actions.index(direction_key)])

            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def train_agent(self, obs, value, policy_action):
        reward = self.reward_pos if obs.reward > 0 else self.reward_neg
        if self.current_state is not None:
            self.sar_batch.append(SAR(self.current_distance, self.current_action, reward))

        if obs.last() or reward > 0:
            reward_sum = np.array()
            train_states = np.array()
            for t, sar in enumerate(self.sar_batch):
                q_val = 0
                if len(self.sar_batch) - t < self.n_step_return:  # n-step is always same size
                    break

                for offset in range(self.n_step_return):    # G
                    if offset == self.n_step_return - 1:
                        q_val += self.gamma ** self.n_step_return * value
                    q_val += self.gamma ** offset * self.sar_batch[t + offset].reward

                advantage = q_val - value
                reward_sum.append([advantage, policy_action])
                train_states.append(sar.state)

            for element in reward_sum:
                element.extend(self.entropy_loss / self.step_count)

            self.network.model.fit(train_states, reward_sum, batch_size=self.mini_batch_size)
            self.new_episode = False

        self.sar_batch = []

    def save_model(self, filename='models'):
        self.network.model.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.network.model.load_weights(self.save)
