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
        self.learning_rate = 0.00025
        self.save = './models/policy_weights_final.h5'
        self.loss = self.loss_function
        self.step_size = 1
        self.model = self.create_model()
        if not train:
            self.load_model()


    def loss_function(self, labels, policy_value_tuple):
        c_val = tf.constant(0.5)
        c_h = tf.constant(0.005)
        advantage, policy_action, entropy = labels[:0], labels[:1], labels[:2]
        policy_loss = tf.mean(tf.stop_gradient(advantage) * np.log(policy_action))
        value_loss = tf.mean(tf.math.square(advantage))
        loss = policy_loss + c_val * value_loss + c_h * entropy
        return loss




    def create_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=8, activation='softmax'))
        # model.build((None, 2))
        model.compile(loss=self.loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        return model


    def load_model(self):
        self.model.load_weights(self.save)

    def save_model(self):
        self.model.save('models')


