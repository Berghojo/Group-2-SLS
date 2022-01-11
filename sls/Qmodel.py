from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys






class Model:
    def __init__(self, input_dim, train, batch_size=32):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.step_size = 1
        if train:
            self.model = self.create_model()
        else:
            self.model = self.load_model()

    def error(self, action_G, policy):
        G, action_index = action_G[:, 0], tf.cast(action_G[:, 1], tf.int32)
        idx = tf.range(0, tf.size(action_index))
        positions = tf.stack([idx, action_index], axis=1)
        policy_action = tf.gather_nd(policy, positions)
        log_policy = tf.math.negative(tf.math.log(policy_action))
        # G = tf.math.negative(G)

        error_t = log_policy * G
        error_sum = tf.math.reduce_sum(error_t)
        error = error_sum / tf.cast(tf.size(G), tf.float32)
        return error

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=8, activation='softmax'))
        # model.build((None, 2))
        model.compile(loss=self.loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        return model



    def load_model(self):
        model = keras.models.load_model('models')
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model




    def save_model(self):
        self.model.save('models')






