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


class Environment:
    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_step, self.m_step = self.stock_price_history.shape
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.action_space = np.arange(3 ** self.m_step)
        self.state_dim = self.m_step * 2 + 1
        self.reset()

    def reset(self):
        self.t = 0
        self.investment = self.initial_investment
        self.stock_owned = 0
        self.history = [(0, 0)] * (self.n_step + 1)
        self.max_portfolio_value = 0

    def step(self, action):
        next_state = self.get_state()

        if action == 1:
            if self.investment >= self.stock_price:
                self.investment -= self.stock_price
                self.stock_owned += 1
        elif action == 2 and self.stock_owned > 0:
            self.investment += self.stock_price
            self.stock_owned -= 1

        self.t += 1
        self.stock_price = self.stock_price_history[self.t, self.m_step - 1]
        portfolio_value = self.investment + self.stock_owned * self.stock_price

        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value

        reward = portfolio_value - self.history[-1][1]
        self.history.append((self.t, portfolio_value))
        done = self.t == self.n_step

        return next_state, reward, done

    def get_state(self):
        state = []
        for i in range(self.m_step):
            state.append(self.stock_price_history[self.t, i])
        return np.array([state])

    def show_chart(self):
        history = np.array(self.history)
        plt.plot(history[:, 0], history[:, 1])
        plt.title('Total money: ${}'.format(self.history[-1][1]))
        plt.show()


class TradingStrategy:

    def __init__(self, model, initial_money, max_buy, max_sell, skip, window_size, vix):
        self.window_size = window_size
        self.skip = skip
        self.model = model
        self.initial_money = initial_money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.get_state = lambda data, t, n: np.array([data[t - i] for i in range(n)])
        self.vix = vix # VIX index data
        # Initialize empty list to store performance data
        self.performance_data = []

    def calcOrderFlow(self, data, t, n):
        '''
        calculates the order flow for the past n days
        '''
        return np.sum([data[t - i] for i in range(n)])

    def calcOrderFlowVolatility(self, data, t, n):
        '''
        calculates the order flow volatility for the past n days
        '''
        return np.std([data[t - i] for i in range(n)])

    def calcVolatility(self, data, t, n):
        return np.std([data[t - i] for i in range(n)])
    
    def calcFUD(self, data, t, n):
        '''
        quantifies the fear, uncertainty, and doubt in the market
        does this by summing the Order Flow Volatility and Stock Volatility 
        multiplying by the change in VIX and change in Stock Volatility
        then dividing by the VIX
        want the value normalized between 0-1
        '''
        order_flow_volatility = self.calcOrderFlowVolatility(data, t, n)
        stock_volatility = self.calcVolatility(data, t, n) 
        change_in_vix = self.vix[t] - self.vix[t - 1] / self.vix[t - 1]
        change_in_stock_volatility = stock_volatility - self.calcVolatility(data, t - 1, n) / self.calcVolatility(data, t - 1, n)
        return ((order_flow_volatility + stock_volatility) * change_in_vix * change_in_stock_volatility) / self.vix[t]

    def moving_average(data, n):
    # Define a variable to store the moving average and initialize it to an empty list
        ma = []

        # Loop through the data series, starting at the first data point
        for i in range(len(data)):
            # Calculate the average of the last n data points
            average = sum(data[i-n:i]) / n
            # Append the calculated average to the moving average list
            ma.append(average)

        # Return the moving average list
        return ma 
    def stop_loss(self, data):
        '''
        Function to create a stop loss at 10% so any stock that loses 10%
        of its value is sold to prevent further losses
        '''
        # Calculate the stop loss price
        stop_loss_price = data * 0.9
        return stop_loss_price
        


    def act(self, data):
        '''
        Algo is more aggressive with high FUD values
        '''

        decision, buy = self.model.predict(np.array(sequence))
        # Calculate penalties for maximum drawdown and sortino ratio
        initial_money = self.initial_money
        starting_money = initial_money
        inventory = []
        quantity = 0
        T = len(data)
        max_drawdown = 0
        max_portfolio_value = 0
        sortino_ratio = 0
        for t in range(0, T, self.skip):
            month_long_ma = self.moving_average(data[t], 30)
            current_vix = self.vix[t]
            fud = self.calcFUD(data, t, self.window_size)
            action = np.argmax(decision[0])
            next_state = self.get_state(data, t + 1, self.window_size + 1)
            if month_long_ma > data[t]:
                # stock is over sold
                print('over sold should buy')
            else:
                print('not over sold')


            if current_vix > 25:
                max_buy_units = self.max_buy / 2 
            else:
                max_buy_units = self.max_buy
            if fud > 0.5:
                print('high fud')
            else:
                print('low fud')

            if action == 1 and initial_money >= data[t]:
                if buy < 0:
                    buy = 1
                if buy > max_buy_units:
                    buy_units = max_buy_units
                else:
                    buy_units = buy
                total_buy = buy_units * data[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * data[t]
                initial_money += total_sell

            portfolio_value = initial_money + sum(inventory)
            if portfolio_value > max_portfolio_value:
                max_portfolio_value = portfolio_value
            else:
                drawdown = (portfolio_value - max_portfolio_value) / max_portfolio_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            daily_return = initial_money / starting_money - 1
            downside_return = daily_return if daily_return < 0 else 0
            downside_risk = np.sqrt(np.mean(downside_return ** 2))
            expected_return = np.mean(daily_return)
            if downside_risk == 0:
                sortino_ratio = expected_return
            else:
                sortino_ratio = expected_return / downside_risk

            # Store performance data for this time step
            performance_data = {
                "timestep": t,
                "initial_money": initial_money,
                "portfolio_value": portfolio_value
            }
            self.performance_data.append(performance_data)

        return action, buy, max_drawdown, sortino_ratio

    
    def logging(self, action, buy, max_drawdown, sortino_ratio):
        # Create a log file and add the results to a csv file
        # Using pandss to create a dataframe and then export to csv
        performance = pd.DataFrame(self.performance_data)
        df = pd.DataFrame({'Action': action, 'Buy': buy, 'Max Drawdown': max_drawdown, 'Sortino Ratio': sortino_ratio})
        df.to_csv('log.csv', mode='a', header=False)
        print('-------------------------')
        if action == 1:
            print('Buy: {}'.format(buy))
        elif action == 2:
            print('Sell: {}'.format(buy))
        print('Max Drawdown: {}'.format(max_drawdown))
        print('Sortino Ratio: {}'.format(sortino_ratio))
        print('-------------------------')


        
    def run(self, data):
        state = self.get_state(data, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        T = len(data) - 1
        initial_money = self.initial_money
        starting_money = initial_money
        for t in range(0, T, self.skip):
            action, buy, max_drawdown, sortino_ratio = self.act(state)
            self.logging(action, buy, max_drawdown, sortino_ratio)
            next_state = self.get_state(data, t + 1, self.window_size + 1)
            if action == 1 and initial_money >= data[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * data[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                print('Buy: {}'.format(total_buy))
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * data[t]
                initial_money += total_sell
                print('Sell: {}'.format(total_sell))
            else:
                print('Hold')
            state = next_state
        print('Total Profit: {}'.format(initial_money - starting_money))
        print('Total Profit Percentage: {}'.format((initial_money - starting_money) / starting_money))
    
  


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