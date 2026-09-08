# Project 09: Algorithmic Trading Reinforcement Learning

## 1. Problem Statement & Business Context
Financial asset markets exhibit dynamic regime shifts (Bullish expansions, Bearish sell-offs, Sideways chop). Static rule-based trading indicators (e.g. fixed RSI < 30) fail when volatility regimes change.

This project implements an **Algorithmic Trading Bot powered by Reinforcement Learning (Q-Learning)**:
- Formulates trading as a **Markov Decision Process (MDP)** across 3 market regimes: Bearish (0), Neutral (1), Bullish (2).
- Learns optimal action policies across 3 actions: HOLD (0), BUY (1), SELL (2) to maximize cumulative portfolio rewards while managing drawdowns.

---

## 2. System Architecture
```
                     [ Historical Price Series ]
                                  │
                                  ▼
                   [ State Discretization Engine ]
                 (Fast/Slow Moving Average Spread)
                   State 0: Bearish (Spread < -0.5)
                   State 1: Neutral (-0.5 <= Spread <= 0.5)
                   State 2: Bullish (Spread > 0.5)
                                  │
                                  ▼
                     [ Q-Learning Agent Environment ]
                Bellman Update: Q(s,a) <- Q + alpha[r + gamma*max Q' - Q]
                                  │
                                  ▼
                      [ Learned Policy Matrix Q* ]
                                  │
                                  ▼
                    [ Live Order Execution Gateway ]
                   Instantaneous HOLD / BUY / SELL Orders
```

---

## 3. Mathematical Formulation
### The Bellman Optimality Equation
For state $s \in \{0, 1, 2\}$ and action $a \in \{0, 1, 2\}$, learning rate $\alpha = 0.10$, discount factor $\gamma = 0.95$:

$$Q_{t+1}(s, a) = Q_t(s, a) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s, a) \right]$$

Where reward $r$ is the realized percentage gain on liquidated positions or holding gains.

---

## 4. Project Structure & Components
```
projects/09_algorithmic_trading_reinforcement_learning/
├── 01_algorithmic_trading_reinforcement_learning_masterclass.ipynb  # Masterclass notebook
├── README.md                                                        # Comprehensive documentation
├── trading_env.py                                                   # Market Environment & Q-Learning Agent
└── test_trading.py                                                  # Pytest verification suite
```

---

## 5. Masterclass Notebook Walkthrough
1. **Problem Statement & Market Regimes**: Defining the sequential decision trading challenge.
2. **Price History Ingestion & State Engineering**: Discretizing momentum spreads into 3 balanced states.
3. **Q-Learning Simulation & Training**: Running 200 episodes with epsilon-greedy decay and reward curves.
4. **Policy Checkpointing & Live Order Execution**: Saving Q-table to `models/algorithmic_trading_rl_model.joblib`.
5. **Executive Summary & Risk Governance**: Stop-loss guardrails (-2.5%), Sharpe ratio, and execution speed.

---

## 6. Running Production Microservices
```bash
# 1. Run unit tests
pytest projects/09_algorithmic_trading_reinforcement_learning/test_trading.py

# 2. Run trading environment module
python projects/09_algorithmic_trading_reinforcement_learning/trading_env.py
```

---

## 7. Performance Benchmarks & SLAs
- **Order Decision Latency**: < 0.05 milliseconds per market tick.
- **Episode Reward Convergence**: Steady progression from negative initial returns to > +50% cumulative return.
- **Policy Behavior**: Dominant action is BUY (1) in Bullish states and SELL (2) in Bearish states.
