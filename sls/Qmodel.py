from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import sys


class Model:
    def __init__(self, input_dim, train, batch_size=32, is_duel=False):
        self.batch_size = batch_size
        self.input_dim = input_dim
        if not is_duel:
            self.model = self.create_model()
        elif is_duel:
            self.model = self.create_lambda_model()

    def lmd(self, input):
        v = input[:,0]  # shape 32
        v = tf.reshape(v, [-1, 1])
        a = input[:,1:] # shape 32,8
        a_mean = K.mean(a, axis=1)
        a_mean = tf.reshape(a_mean, [-1, 1])
        Q = v + (a - a_mean)
        return Q

    def create_lambda_model(self):
        # base model
        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=9, activation='linear'))
        # lambda layer
        # [V, A1, A2, ..., A8]
        model.add(Lambda(self.lmd))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='linear'))
        model.build((None, 2))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model

    def load_model(self):
        model = keras.models.load_model('models')
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model

    def save_model(self):
        self.model.save('models')
