"""
Reinforcement Learning Trading Environment.
"""

import numpy as np

class TradingEnvironment:
    def __init__(self, prices: np.ndarray, initial_balance: float = 10000.0):
        self.prices = prices
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.balance = self.initial_balance
        self.shares = 0
        return self._get_state()

    def _get_state(self):
        current_price = self.prices[self.step_idx]
        return np.array([current_price, self.balance, self.shares], dtype=np.float32)

    def step(self, action: int):
        # 0: Hold, 1: Buy, 2: Sell
        price = self.prices[self.step_idx]
        if action == 1 and self.balance >= price: # Buy
            self.shares += 1
            self.balance -= price
        elif action == 2 and self.shares > 0: # Sell
            self.shares -= 1
            self.balance += price

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1
        portfolio_val = self.balance + (self.shares * price)
        reward = portfolio_val - self.initial_balance
        next_state = self._get_state() if not done else np.zeros(3)
        return next_state, reward, done, {"portfolio_value": portfolio_val}
