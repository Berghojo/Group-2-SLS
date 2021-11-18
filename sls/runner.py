import datetime
import os
import tensorflow as tf
import numpy as np


class Runner:
    def __init__(self, agent, env, train, load_path):

        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?
        self.scores_batch = []
        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter

        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        # self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        self.writer = tf.summary.create_file_writer(self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        with self.writer.as_default():
            tf.summary.scalar('Epsilon per Episode', self.agent.get_epsilon(), step=self.episode)
            tf.summary.scalar('Score per Episode', self.score, step=self.episode)
            self.scores_batch.append(self.score)
            if len(self.scores_batch) == 50:
                average = np.mean(self.scores_batch)
                tf.summary.scalar('Moving Average Score (50) per Episode', average, step=self.episode - 50)
                self.scores_batch.pop(0)

        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
        self.episode += 1
        self.score = 0

    def run(self, episodes):
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            self.summarize()
            print(self.agent.temperature_decay(self.episode))

