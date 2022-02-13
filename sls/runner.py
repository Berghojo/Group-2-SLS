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
        self.worker_handles = None
        self.writer = tf.summary.create_file_writer(self.path)
        #self.writer = tf.compat.v1.summary.FileWriter(self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.network.load_model(load_path)

    def summarize(self):
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

            self.sar_batch = []
            self.entropy_loss = 0
            for parent, _ in self.pipes:
                parent.send('reset')
            not_new_episode = True
            while not_new_episode:
                for parent, _ in self.pipes:
                    parent.send('step')

                states = []
                for i, (parent, _) in enumerate(self.pipes):  # Send predictions to workers
                    state = parent.recv()
                    states.append(state)
                prediction = self.network.model.predict(np.array(states).reshape([-1, 16, 16, 1]), verbose=0)
                for i, (parent, _) in enumerate(self.pipes):

                    parent.send(prediction[i])
                    policy_probability = prediction[i][:-1]

                    if any(policy_probability) <= 0:
                        entropy = 0
                    else:
                        entropy = -np.sum(policy_probability * np.log(policy_probability))
                    self.entropy_loss += entropy

                for parent, _ in self.pipes:    # Check worker status
                    parent.send('check')

                for i, (parent, _) in enumerate(self.pipes):  # Recv message from workers
                    agent_response = parent.recv()

                    if isinstance(agent_response, tuple):   # Agents finished one game
                        agent_sar_batch, reward = agent_response
                        self.sar_batch += agent_sar_batch
                        self.score += reward

                    elif isinstance(agent_response, int):    # If worker is finished
                        if not agent_response in finished_agents:   # New agents finished
                            finished_agents.append(agent_response)
                            agent_sar_batch = parent.recv()
                            self.sar_batch += agent_sar_batch
                        else:   # Already finished agents (discard message)
                            parent.recv()
                        if len(finished_agents) == self.num_workers:    # All agents finished sync
                            self.train_network()
                            self.summarize()
                            not_new_episode = False


        for parent, _ in self.pipes:
            parent.send('close')
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
        self.entropy_loss = 0
        train_states = np.array(train_states)

        reward_sum = np.array(reward_sum)
        self.network.model.fit(train_states, y=reward_sum, batch_size=16, verbose=1)
        self.sar_batch = []


class Worker:

    def __init__(self, env_func, screen_size, train, agent, conn, id):
        self.id = id
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
        self.done = False

    def run_worker(self):
        self.env = self.env_func()
        obs = self.env.reset()
        while self.running:
            command = self.conn.recv()
            # Recv block
            if command == 'step':
                current_state = obs.observation.feature_screen['unit_density'].reshape([16, 16, 1])
                self.conn.send(current_state)
                prediction = self.conn.recv()
                if not obs.last():
                    action = self.agent.step(obs, prediction)
                    obs = self.env.step(action)

            if command == "check":
                if obs.last():  # Waiting
                    self.conn.send(self.id)
                    self.conn.send((self.agent.sar_batch))
                    self.done = True
                elif obs.reward > 0:
                    self.conn.send((self.agent.sar_batch, obs.reward))
                    self.agent.sar_batch = []
                else:
                    self.conn.send('Nothing')
            if command == 'reset':
                self.agent.sar_batch = []
                obs = self.env.reset()

            if command == 'close':
                self.running = False
                self.conn.close()
