from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, InputLayer, Conv2D, Softmax
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

    def error(self, labels, policy_value):
        c_val = tf.constant(0.5)
        c_h = tf.constant(0.005)
        value = policy_value[:, 8]
        policy = policy_value[:, :-1]
        q_val, action_index, entropy = labels[:, 0], tf.cast(labels[:, 1], tf.int32), labels[:, 2]
        idx = tf.range(0, tf.size(action_index))
        advantage = q_val - value
        positions = tf.stack([idx, action_index], axis=1)
        policy_action = tf.gather_nd(policy, positions)
        policy_loss = tf.math.negative(tf.math.reduce_mean(tf.stop_gradient(advantage) * tf.math.log(policy_action)))
        value_loss = tf.math.reduce_mean(tf.math.square(advantage))
        loss = policy_loss + c_val * value_loss + c_h * entropy
        return loss

    def create_model(self):
        inputs = Input(shape=(16, 16, 1), name="img")
        l1 = Conv2D(16, (5, 5), strides=1, padding="same", activation="relu")(inputs)
        l2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(l1)
        actor = Conv2D(1, (1, 1), strides=1, padding="same", activation="linear", name="actor_out")(l2)
        l3 = Flatten()(actor)
        actor_flatten = Softmax(name='softmax_actor')(l3)
        x = Dense(256, activation="relu")(actor_flatten)
        critic = Dense(1, activation="linear", name="criticout")(x)
        prediction = [tf.concat([actor_flatten, critic], 1)]
        model = Model(inputs=inputs,
                      outputs=prediction,
                      name='A2C')

        model.compile(loss=self.loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        # model.summary()
        return model

    def load_model(self):
        self.model.load_weights(self.save)

    def save_model(self):
        self.model.save('models')






