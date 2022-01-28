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
        self.learning_rate = 0.0007
        self.save = './models/A2C_weights_final.h5'
        self.loss = self.error
        self.step_size = 1
        self.model = self.create_model()
        if not train:
            self.load_model()

    def error(self, labels, policy_values):
        c_val = tf.constant(0.5)
        c_h = tf.constant(0.005)
        advantage, policy_action, entropy = labels[:0], labels[:1], labels[:2]
        policy_loss = tf.math.reduce_mean(tf.stop_gradient(advantage) * tf.math.log(policy_action))
        value_loss = tf.math.reduce_mean(tf.math.square(advantage))
        loss = policy_loss + c_val * value_loss + c_h * entropy
        return loss

    def create_model(self):
        inputs = Input(shape=(16, 16, 1), name="img")
        x = Conv2D(16, 5, strides=1, padding="same", activation="relu")(inputs)
        x = Conv2D(32, 3, strides=1, padding="same", activation="relu")(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        actor = Dense(8, activation="softmax")(x)
        critic = Dense(1, activation="linear")(x)
        model = Model(inputs=inputs,
                      outputs=[actor, critic],
                      name='A2C')

        model.summary()
        model.compile(loss=self.loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        return model

    def load_model(self):
        self.model.load_weights(self.save)

    def save_model(self):
        self.model.save('models')






