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
        if train:
            self.model = self.create_model()
        else:
            self.model = self.load_model()

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


