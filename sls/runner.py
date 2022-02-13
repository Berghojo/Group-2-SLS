import datetime
import gc
import os
import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pipe
import tracemalloc

class Runner:
    def __init__(self, agent, train, load_path, config):

        self.agent = agent
        self.config = config
        self.train = train  # run only or train_model model?
        self.scores_batch = []
        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter
        self.current_average = 0
        self.num_workers = 2
        self.worker_processes = []
        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        # self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        self.writer = tf.compat.v1.summary.FileWriter(self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):

        self.scores_batch.append(self.score)
        if len(self.scores_batch) == 50:
            average = np.mean(self.scores_batch)
            self.writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='Moving Average Score (50) per Episode', simple_value=average)]), self.episode - 50)
            self.scores_batch.pop(0)
            self.current_average = average
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
        self.episode += 1
        self.score = 0

    def run(self, episodes, env_func):

        self.start_workers(env_func)


    def start_workers(self, env_function):
        worker_handles = [Worker(env_function, 1) for _ in range(self.num_workers)]
        print('the what')
        self.worker_processes = [Process(target=worker_handle.run_worker) for worker_handle in worker_handles]
        print(self.worker_processes)
        for process in self.worker_processes:
            process.start()

class Worker:

    def __init__(self, env_func, agent):
        self.env_func = env_func
        self.episode = 0
        self.agent = agent
        self.env = None
        self.score = 0

    def run_worker(self):
        episodes = 1000
        self.env = self.env_func()
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            #self.summarize()
            #print('Average: ', self.current_average)
