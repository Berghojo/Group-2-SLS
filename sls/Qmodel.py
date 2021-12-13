from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Lambda, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import sys


class Model:
    def __init__(self, input_dim, train, batch_size=32, is_duel=False, is_conv=False, is_conv_2=False):
        self.batch_size = batch_size
        self.input_dim = input_dim
        if is_duel:
            self.model = self.create_lambda_model()
        elif is_conv:
            self.model = self.create_convolutional_model()
        elif is_conv_2:
            self.model = self.create_convolutional_model_2()
        else:
            self.model = self.create_model()

    def lmd(self, input):
        v = input[:,0]  # shape 32
        v = tf.reshape(v, [-1, 1])
        a = input[:,1:] # shape 32,8
        a_mean = tf.math.reduce_mean(a, axis=1)
        a_mean = tf.reshape(a_mean, [-1, 1])
        Q = v + (a - a_mean)
        return Q

    def create_convolutional_model(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, activation='relu', strides=1, input_shape=(16, 16, 1),
                         padding='same',
                         kernel_initializer='he_normal'))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same',
                         kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(9, activation='linear', kernel_initializer='he_normal'))
        model.add(Lambda(self.lmd))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model

    def create_convolutional_model_2(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, activation='relu', strides=1, input_shape=(16, 16, 1),
                         padding='same',
                         kernel_initializer='he_normal'))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same',
                         kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(8, activation='linear', kernel_initializer='he_normal'))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        return model

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
