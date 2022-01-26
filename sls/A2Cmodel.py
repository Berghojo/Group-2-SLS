from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys


class A2CModel:
    def __init__(self, input_dim, train, batch_size=32):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.learning_rate = 0.00025
        self.save = './models/A2C_weights_final.h5'
        self.loss = self.error
        self.step_size = 1
        self.model = self.create_model()
        if not train:
            self.load_model()

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
        inputs = Input(shape=(16, 16, 1), name="img")
        x = Conv2D(16, 5, strides=1, padding="same", activation="relu")(inputs)
        x = Conv2D(32, 3, strides=1, padding="same", activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        actor = Dense(8, activation="softmax")(x)
        critic = Dense(1, activation="linear")(x)
        model = Model(inputs=inputs,
                      outputs=[actor, critic],
                      name='A2C')

        model.summary()
        return model

    def load_model(self):
        self.model.load_weights(self.save)

    def save_model(self):
        self.model.save('models')






