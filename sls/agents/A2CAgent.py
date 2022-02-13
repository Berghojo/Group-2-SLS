import sys
from sls import helper
from sls.agents import AbstractAgent
import numpy as np
from sls.A2Cmodel import A2CModel
from multiprocessing.managers import BaseManager
import tensorflow as tf
from multiprocessing import Lock
from collections import namedtuple

SAR = namedtuple("SAR", ["state", "action", "reward", "value"])


class A2CAgent(AbstractAgent):
    def __init__(self, screen_size, train):
        id = 1
        print(f'Created Agent {id}')
        super(A2CAgent, self).__init__(screen_size)
        self.save = f'./models/A2C_weights_final_{id}.h5'
        self.actions = list(self._DIRECTIONS.keys())
        self.pos_reward = 1
        self.neg_reward = -0.01
        self.mini_batch_size = 64
        self.learning_rate = 0.0007
        self.entropy_constant = 0.005
        self.value_constant = 0.5
        self.n_step_return = 5
        self.sar_batch = []
        self.current_state = None
        self.current_action = None
        self.new_episode = True
        self.step_count = 1
        self.entropy_loss = 0
        self.last_value = None
        self.gamma = 0.99

    def step(self, obs, prediction):
        self.step_count += 1
        if self._MOVE_SCREEN.id in obs.observation.available_actions:

            reward = self.pos_reward if obs.reward > 0 else self.neg_reward
            if self.current_state is not None:
                self.sar_batch.append(SAR(self.current_state, self.current_action, reward, self.last_value))

            marine = self._get_marine(obs)
            beacon_object = self._get_beacon(obs)

            if beacon_object is None or marine is None:
                return self._NO_OP

            marine_coordinates = self._get_unit_pos(marine)

            self.current_state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
            self.current_state = np.array(self.current_state)
            policy_probability, value = prediction[:-1], prediction[8]
            self.last_value = value
            policy_probability = np.squeeze(policy_probability)
            direction_key = np.random.choice(self.actions, p=policy_probability)
            self.current_action = direction_key
            #if self.train:
            #    self.train_agent(obs, value, policy_probability[self.actions.index(direction_key)])

            return self._dir_to_sc2_action(direction_key, marine_coordinates)
        else:
            return self._SELECT_ARMY

    def train_agent(self, obs):

        if obs.last() or reward > 0:
            reward_sum = []
            train_states = []
            'calculating values'

            for t, sar in enumerate(self.sar_batch):
                q_val = 0

                for offset in range(self.n_step_return):    # G
                    q_val += self.gamma ** offset * self.sar_batch[t + offset].reward
                    if t + offset >= len(self.sar_batch)-1:
                        break
                    elif offset == self.n_step_return-1:
                        q_val += self.gamma ** self.n_step_return * self.sar_batch[t + self.n_step_return].value

                reward_sum.append([q_val, self.actions.index(sar.action)])
                train_states.append(sar.state)
            entropy_loss = -(self.entropy_loss / len(reward_sum))
            for element in reward_sum:
                element.append(entropy_loss) # entropy_loss mean
            train_states = np.array(train_states)
            reward_sum = np.array(reward_sum)
            self.network.model.fit(train_states, y=reward_sum, batch_size=self.mini_batch_size)
            self.current_state = None
            self.sar_batch = []

    def save_model(self, filename='models'):
        self.network.model.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.network.model.load_weights(self.save)


class KerasManager(BaseManager):
    pass
