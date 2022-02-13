import datetime
import gc
import os
import tensorflow as tf
from sls.A2Cmodel import A2CModel
import numpy as np
from sls.agents import *
from multiprocessing import Process, Pipe, Lock
import tracemalloc


class Runner:
    def __init__(self, agent, train, load_path, config):

        self.entropy_loss = 0
        self.actions = None
        self.mini_batch_size = 16
        self.actions = list(agent._DIRECTIONS.keys())
        self.agent = agent
        self.config = config
        self.train = train  # run only or train_model model?
        self.scores_batch = []
        self.score = 10  # store for the scores of an episode
        self.episode = 1  # episode counter
        self.mutex = Lock()
        self.current_average = 0
        self.num_workers = 8
        self.worker_processes = []
        self.sar_batch = []
        self.n_step_return = 5
        self.gamma = 0.99
        self.network = A2CModel(True)
        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__
        self.pipes = []
        self.writer = tf.summary.create_file_writer(self.path)
        #self.writer = tf.compat.v1.summary.FileWriter(self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.network.load_model(load_path)

    def summarize(self):

        self.scores_batch.append(self.score)
        with self.writer.as_default():
            average = np.mean(self.score / self.num_workers)
            tf.summary.scalar('Average Score of Workers per episode', average, step=self.episode)
        self.current_average = average
        print('average:', self.current_average)
        if self.train and self.episode % 10 == 0:
            self.network.save_model(self.path)
        self.episode += 1
        self.score = 0

    def run(self, episodes, env_func):
        self.start_workers(env_func)
        while self.episode <= episodes:
            finished_workers = 0
            self.sar_batch = []
            for parent, _ in self.pipes:
                parent.send('reset')
            not_new_episode = True
            while not_new_episode:
                for parent, _ in self.pipes:
                    parent.send('step')
                states = []

                for parent, _ in self.pipes:
                    states.append(parent.recv())

                prediction = self.network.model.predict(np.array(states).reshape([-1, 16, 16, 1]), verbose = 0)

                i = 0
                for parent, _ in self.pipes:
                    parent.send(prediction[i])
                    policy_probability = prediction[i][:-1]

                    if any(policy_probability) <= 0:
                        print('ZERO')
                        entropy = 0
                    else:
                        entropy = -np.sum(policy_probability * np.log(policy_probability))
                    self.entropy_loss += entropy
                    i += 1
                worker_id = 0
                for parent, _ in self.pipes:
                    msg = parent.recv()
                    worker_id += 1
                    if msg == 'done':
                        finished_workers += 1
                        with self.mutex:
                            new_batch = parent.recv()
                            print(len(new_batch))
                            self.sar_batch = self.sar_batch + new_batch
                    elif msg != 'waiting':
                        self.score += msg

                if finished_workers == self.num_workers:
                    with self.mutex:
                        self.train_network()
                    self.summarize()
                    break

        for process in self.worker_processes:
            process.join()


    def start_workers(self, env_function):
        self.pipes = [Pipe() for _ in range(self.num_workers)]
        worker_handles = [Worker(env_function, self.config['screen_size'], self.config['train'], self.agent, self.pipes[i][1]) for i in
                          range(self.num_workers)]
        self.worker_processes = [Process(target=worker_handle.run_worker) for worker_handle in worker_handles]
        print(self.worker_processes)
        for process in self.worker_processes:
            process.start()

    def train_network(self):
        reward_sum = []
        train_states = []
        'calculating values'

        for t, sar in enumerate(self.sar_batch):
            q_val = 0

            for offset in range(self.n_step_return):  # G
                q_val += self.gamma ** offset * self.sar_batch[t + offset].reward
                if t + offset >= len(self.sar_batch) - 1:
                    break
                elif offset == self.n_step_return - 1:
                    q_val += self.gamma ** self.n_step_return * self.sar_batch[
                        t + self.n_step_return].value

            reward_sum.append([q_val, self.actions.index(sar.action)])
            train_states.append(sar.state)

        entropy_loss = -(self.entropy_loss / len(reward_sum))
        for element in reward_sum:
            element.append(entropy_loss)  # entropy_loss mean
        train_states = np.array(train_states)
        reward_sum = np.array(reward_sum)
        self.network.model.fit(train_states, y=reward_sum, batch_size=16, verbose=1)
        self.current_state = None
        self.sar_batch = []


class Worker:

    def __init__(self, env_func, screen_size, train, agent, conn):
        self.env_func = env_func
        self.episode = 0
        self.screen_size = screen_size
        self.train = train
        self.agent = agent
        self.env = None
        self.done = False
        self.score = 0
        self.conn = conn
        self.running = True

    def run_worker(self):
        self.env = self.env_func()
        obs = self.env.reset()
        while self.running:
            command = self.conn.recv()
            if command == 'reset':
                obs = self.env.reset()
                self.agent.sar_batch = []
                self.done = False
            if command == 'step':
                current_state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
                self.conn.send(current_state)
                prediction = self.conn.recv()

                if (obs.last() or obs.reward > 0) and not self.done:
                    self.conn.send('done')
                    self.conn.send(self.agent.sar_batch)

                    self.done = True
                elif not self.done:
                    action = self.agent.step(obs, prediction)
                    obs = self.env.step(action)
                    self.conn.send(obs.reward)
                else:
                    self.conn.send('waiting')
            #while True:
            #    current_state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
            #    action = self.agent.step(obs)
            #    if obs.last():
            #        break
            #    obs = self.env.step(action)
            #    self.conn.send('Done')
            #    self.score += obs.reward
            # self.summarize()
            # print('Average: ', self.current_average)
