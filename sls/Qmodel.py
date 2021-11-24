from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import RMSprop
import numpy as np


class Model:
    def __init__(self, input_dim, batch_size=32):
        self.epsilon_start = 1
        self.epsilon_min = 0.05
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
        return model

    def save_model(self):
        self.model.save('models')


