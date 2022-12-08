import numpy as np
import random
import pandas as pd
import tensorflow as tf

from collections import deque
from typing import List, Tuple

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K


class DeepQLearning:
    def __init__(self, state_size: int, action_size: int, optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), replay_buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size

        # Define the memory attribute as a property
        self._memory = deque(maxlen=replay_buffer_size)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.optimizer = optimizer 
        self.model = self._build_model()
        self.target_model = self._build_model()

        self.update_target_model()

    @property
    def memory(self) -> List[Tuple]:
        return self._memory

    @memory.setter
    def memory(self, memory: List[Tuple]):
        self._memory = memory

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        linear_loss = clip_delta * (K.abs(error) - 0.5 * clip_delta)

        return K.mean(tf.where(cond, squared_loss, linear_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=self.optimizer(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)

        # Use the isclose function to check whether the exploration rate is close
        # to the minimum exploration rate
        if np.isclose(self.epsilon, self.epsilon_min):
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Use the "with" keyword to ensure the file is properly closed after
        # being read
        with open(name, 'rb') as f:
            self.model.load_weights(f)

    def save(self, name):
        # Use the "with" keyword to ensure the file is properly closed after
        # being written
        with open(name, 'wb') as f:
            self.model.save_weights(f)



class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights = [
            np.random.randn(input_dim, hidden_dim),
            np.random.randn(hidden_dim, output_dim),
            np.random.randn(hidden_dim, 1),
            np.random.randn(1, hidden_dim),
        ]
    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, trading_strategy):
        self.state_size = state_size
        self.action_size = action_size
        self.trading_strategy = trading_strategy

        self.dqn_model = DeepQLearning(state_size, action_size)
        super().__init__(state_size, action_size, self.dqn_model.model)

    def act(self, state, max_drawdown=0, sortino_ratio=0):
        if np.random.rand() <= self.dqn_model.epsilon:
            return random.randrange(self.action_size)

        action, buy, penalty_drawdown, penalty_sortino = self.trading_strategy.act(state)
        if penalty_drawdown > max_drawdown or penalty_sortino < sortino_ratio:
            return 0  # Hold action
        if buy:
            return 1  # Buy action
        return action  # Use deep Q-learning model to select action




    def update(self):
        self.dqn_model.replay(32)
        super().update()

    def save(self, name):
        self.dqn_model.save(name)

    def load(self, name):
        self.dqn_model.load(name)

    def update_target_model(self):
        self.dqn_model.update_target_model()

    def memorize(self, state, action, reward, next_state, done):
        self.dqn_model.memorize(state, action, reward, next_state, done)
