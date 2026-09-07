"""
Tests for RL Trading Environment.
"""

import numpy as np
from trading_env import TradingEnvironment

def test_trading_env_step():
    prices = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
    env = TradingEnvironment(prices)
    state = env.reset()
    assert state[0] == 100.0
    
    next_s, reward, done, info = env.step(action=1) # Buy
    assert env.shares == 1
    assert not done
