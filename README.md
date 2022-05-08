```csharp
pip install -r requirements.txt
```
# Reinforcement Learning Implementation- Grid World

This project is meant to demonstrate a wide variety of RL algorithms in Grid World. Including Dynamic Programming : Value iterations， Policy iteration Model-free: MC，Q-learning, SARSA, Policy Gradient.


* **`main_.py`** - Just run it to view different algorithms.

* **`agent_.py`and`grid_env.py`** - Defines different agents for different algorithms and grid world for traditional algorithms(grid_env.py: actions that lead to forbidden areas or the boundaries are excluded) and DRL(grid_env_nn.py: included)

* **`StoredTrainingData`** - Trained deep neural network and V, Q Tabs.

## For assignment 1 
Include main_DyPro-v.py,  Agent_DP_V.py, grid_DP_v.py.

To see the two derterministic policy, 

Run main_DyPro-v.py with
```csharp
#chosenAction=agent.policyAction(env.state,epsilon)
chosenAction=agent.deterministicPolicy(env.state,epsilon)
```
## For assignment 2
Include Assignment2.py,  Agent_DP_V.py, grid_DP_v.py；
Run Assignment2.py

## For assignment 3
Include main_DyPro-v.py,  Agent_DP_V.py, grid_DP_v.py；

To see Value iteration of Dynamic programming,

Run main_DyPro-v.py with
```csharp
chosenAction=agent.policyAction(env.state,epsilon)
#chosenAction=agent.deterministicPolicy(env.state,epsilon)
```