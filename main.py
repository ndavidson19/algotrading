import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enviroment import Environment
from deepqlearning import DQNAgent
from strategies.stocks import TradingStrategy

if __name__ == '__main__':
    data = np.random.random((300, 5))
    env = Environment(data)
    trading_strategy = TradingStrategy(env)
    state_size = 5
    action_size = 3
    agent = DQNAgent(state_size, action_size, trading_strategy)

    EPISODES = 1000
    for e in range(EPISODES):
        state = env.reset()
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                agent.update()
                break
        if e % 10 == 0:
            agent.update()

    env.show_chart()

    agent.save('dqn.h5')