import sys
import time

import gym
import environment.grid_env_nn as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_DP_p as ag
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


env=grid.GridEnv1_nn()
agent=ag.agent_DP_p()
num_episodes = 300
max_number_of_stepsv=100
max_number_of_stepsp=100
epsilon=0.8
dicrease=0.01
a=0.2   #LearningRate
allrewards=np.zeros(1)
path="StoredTrainingData\\DP_p.pt"

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    if os.path.exists(path):
        agent.Policy.pi=torch.load(path)

    #policy evaluation
    for t in range(max_number_of_stepsv):
        env.render()
        chosenAction,logpro=agent.policyAction(env.state)
        prevstate=env.state

        nextstate, reward, Terminal=env.step(chosenAction)

        agent.Policy.buffer.push(prevstate,chosenAction, reward, nextstate,Terminal)


        e_return+=reward
        if Terminal==True:
            break



    if i%4==0:
        # policy improvement
     agent.Policy.train_net(agent.vtab)

    if i % 29 == 0:
     print('score:',e_return,'\n',i ,'episode\n value:')
     print(agent.vtab)

     agent.Policy.plot_cost()

    allrewards=np.append(allrewards,e_return)

torch.save(agent.Policy.pi,path)

env.close()
x=np.asarray(range(len(allrewards)))
fig=plt.plot(x,allrewards)
plt.savefig("Dypro-p.png")
plt.show()
sys.exit()
