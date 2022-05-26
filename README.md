```csharp
pip install -r requirements.txt
```
# Reinforcement Learning Implementation- Grid World

This project is meant to demonstrate a wide variety of RL algorithms in Grid World. Including Dynamic Programming : Value iterations， Policy iteration Model-free: MC，Q-learning, SARSA, Policy Gradient.


* **`main_.py`** - Just run it to view different algorithms.

* **`agent_.py`and`grid_env.py`** - Defines different agents for different algorithms and grid world for traditional algorithms(grid_env.py: actions that lead to forbidden areas or the boundaries are excluded) and DRL(grid_env_nn.py: included)

* **`StoredTrainingData`** - Trained deep neural network and V, Q Tabs.


## For assignment 4
Include main_PG.py,  agent_PG.py,agent_PG_e_greedy, grid_env_nn.py；

To see MC Exploring Starts,

Run main_PG.py with
```csharp
#import agent.agent_PG as ag
import agent.agent_PG_e_greedy as ag
```
To see MC epsilon-greedy,

Run main_PG.py with
```csharp
import agent.agent_PG as ag
#import agent.agent_PG_e_greedy as ag
```