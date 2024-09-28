# Dependencs

# pip install lumibot==2.9.13

# Easy algo trading framework
from tkinter import YES
from lumibot.brokers import  Alpaca # Alpaca is going to be broker
from lumibot.backtesting import YahooDataBacktesting # Give framework for backtesting
from lumibot.strategies.strategy import Strategy # It can be a actual trading bot
from lumibot.traders import Trader # Give us our deployment capability if you want to 

# For news
from alpaca_trade_api import REST

from timedelta import Timedelta

# Alpaca - trade - api - python :- Get news and place trades to broker

# Datatime :-  Mainly for date formating
from datetime import datetime
# timedelta :- Calculating date difference

# Torch :- pytorch framework for using AI / ML

# Transformers :- Load up finance deep learning model


API_KEY = 'PK2EI3D956LK9CSHIODN'
API_SECRET = 'o4wHHZqWEjwWfRjMl0adCcI28h6B3eKa6qnhyAK7'
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY" : API_KEY,
    "API_SECRET" : API_SECRET,
    "PAPER" : True
}

class MLTrader(Strategy):
    # Positoin sizing our position size is going to be calculated based on a metric called cash at risk  so how much of our cash
    # balance do we want to risk
    def initialize(self, symbol:str='SPY',cash_at_risk:float=.5): # This can be run ones
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        # For News
        self.api = REST(base_url=BASE_URL, key_id= API_KEY , secret_key= API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        
        # The formula guides how much of our cash balance we use per trade. cash_at_risk 0f 0.5 means that for 
        # each trade we are using 50% of our remaining cash balance
        quantity = (cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self):
        today  =  self.get_datetime() # Today with respect to backtest
        three_days_prior = today - three_days_prior(days = 3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strfttime('%Y-%m-%d') 

    def get_news(self):
        today, three_days_prior= self.get_dates()
        news = self.api.get_news(symbol= self.symbol, 
                                 start= three_days_prior , 
                                 end= today)
        
        news = [ev.__dict__["__raw"]["headline"] for ev in news]

        return news

    def on_trading_iteration(self): # It can be iterated again and again
        cash, last_price, quantity = self.position_sizing()

        if cash > last_price:
            if self.last_trade == None:
                news = self.get_news()
                print(news)
                order = self.create_order(
                    self.symbol,
                    quantity, # How many symbol to buy
                    "buy",
                    # type= "market" # Market, Limit
                    type= "bracket",
                    take_profit_price= last_price*1.20, # 20%
                    stop_loss_price= last_price*.95
                )
            self.submit_order(order)
            self.last_trade = "buy"

start_date = datetime(2023,12,15)
end_date = datetime(2023,12,31)


broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(name='mlstrat', broker = broker, parameters= {"symbol":"SPY",
                                                                  "cash_at_risk ": .5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters= {"symbol":"SPY",
                 "cash_at_risk ": .5}
)