import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO
from gym.spaces import Discrete, Box, MultiDiscrete, Dict
import os
os.system("pip install stable_baselines3")
from stable_baselines3.common.env_checker import check_env
import random
#from getdata import get_data
from time import time, sleep
import yfinance as yf

class BitcoinTradingOnline(gym.Env):
    def __init__(self, API_KEY, API_SECRET_KEY, endpoint_url = 'https://paper-api.alpaca.markets'):
        
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET_KEY, endpoint_url, api_version = 'v2')
        self.account = self.alpaca.get_account()
        self.initial_balance = self.account.cash
        self.bitcoins = 0
        self.action_space = Box(low = -0.25, high = 0.25, shape = (1,)) 
        spaces = {
            'Balance' : Box(low = 0, high = 10000, shape = (1,)),
            'Bitcoins' : Box(low = 0, high = 5, shape = (1,)),
            'Open' : Box(low = 0, high = 100000, shape = (30,)),
            'High' : Box(low = 0, high = 100000, shape = (30,)),
            'Low' : Box(low = 0, high = 100000, shape = (30,)),
            'Close': Box(low = 0, high = 100000, shape = (30,)),
            'Adj Close': Box(low = 0, high = 100000, shape = (30,)),
            'Volume': Box(low = 0, high = 99999999, shape = (30,))
        }
        self.observation_space = Dict(spaces)
    
    def get_data(self, 
                 symbol = 'BTC-USD'):
        
        data = yf.download(tickers=symbol, period = '145m', interval = '5m')
        
        state = {}
        self.balance = float(self.alpaca.get_account().cash)
        state['Balance'] = np.array([self.balance])
        state['Bitcoins'] = np.array([self.bitcoins])
        state['Open'] = np.array(data["Open"].tolist())
        state['High'] = np.array(data["High"].tolist())
        state['Low'] = np.array(data["Low"].tolist())
        state['Close'] = np.array(data["Close"].tolist())
        state['Adj Close'] = np.array(data["Adj Close"].tolist())
        state['Volume'] = np.array(data["Volume"].tolist())
        
        return state
        
    
    def reset(self):
        self.initial_balance = self.account.cash
        return self.get_data()
    
    def step(self, action):
        self._take_action(action)
        self.net_worth = float(self.alpaca.get_account().equity)
        reward = self.net_worth
        if self.net_worth <= 0:
            done = True
        else:
            done = False
        sleep(60)
        obs = self.get_data()
        return obs, reward, done, {}
        
    def _take_action(self,action):
        data = self.get_data()
        
        if action[0] > 0:
            amount = self.balance*action[0]
            b_bitcoins = amount/data["Close"].tolist()[-1]
            self.bitcoins += b_bitcoins
            self.alpaca.submit_order(symbol = 'BTCUSD',
                                    qty = b_bitcoins,
                                    side = 'buy',
                                    type = 'market',
                                    time_in_force = 'gtc')
            print('Bought ' + str(b_bitcoins) +' ' + 'bitcoins')

        if action[0] < 0:
            amount = -self.bitcoins*action[0]
            self.bitcoins -= amount
            self.alpaca.submit_order(symbol = 'BTCUSD',
                                    qty = amount,
                                    side = 'sell',
                                    type = 'market',
                                    time_in_force = 'gtc')
            print('Sold ' + str(amount) +' ' + 'bitcoins')
            
        def render(self):
            print('Net Worth: ' + str(self.net_worth))

API_KEY = 'PKB6WCAJBR1TT26YFHRH'
API_SECRET_KEY = '1tCiCfHSNlPDYSW24X2gpi9qvf9mzPKBdKfwwks0'
endpoint_url = 'https://paper-api.alpaca.markets'
env = BitcoinTradingOnline(API_KEY, API_SECRET_KEY)

model = DDPG.load("ddpg_bitcoin_hist")

obs = env.reset()
counter = 0

while True:
    try:
        counter = 0
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
        sleep(301)
    except Exception as e:
        if counter != 0:
            continue
        counter += 1
        print("Error Occured. Proceeding... Error is: " + str(e))