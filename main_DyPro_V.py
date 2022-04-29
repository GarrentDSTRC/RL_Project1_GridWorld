import sys
import time

import gym
import environment.grid_env as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_DP_v as ag


env=grid.GridEnv1()
agent=ag.agent_DP_V()
num_episodes = 200
max_number_of_steps=100
epsilon=0.8
dicrease=0.01
allrewards=np.zeros(1)

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    agent.vtab=np.load("StoredTrainingData\\vtab_DP_v.npy")
    for t in range(max_number_of_steps):
        env.render()
        chosenAction=agent.policyAction(env.state,epsilon)
        nextstate, reward, Terminal=env.step(chosenAction)

        agent.vtab[env.state]=reward+agent.gamma*agent.vtab[nextstate]  #Dynamic Programming -Value iterations

        env.state=nextstate
        e_return+=reward
        if Terminal==True:
            break

        time.sleep(0.02)

    print('score:',e_return,'\n',i ,'episode\n value:',agent.vtab)

    allrewards=np.append(allrewards,e_return)
    np.save("StoredTrainingData\\vtab_DP_v.npy",agent.vtab)


env.close()
x=np.asarray(range(len(allrewards)))
plt.plot(x,allrewards)
plt.savefig("DyPro-v.png")
plt.show()
sys.exit()
