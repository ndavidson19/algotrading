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