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

        decision, buy = self.model.predict(np.array(data))
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
    
  