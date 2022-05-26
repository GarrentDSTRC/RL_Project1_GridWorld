import sys
import time

import gym
import environment.grid_env_nn as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_PPO as ag
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env=grid.GridEnv1_nn()
agent=ag.agent_PPO()
num_episodes = 200
max_number_of_stepsv=100
max_number_of_stepsp=100
epsilon=0.8
dicrease=0.01
a=0.2   #LearningRate
allrewards=np.zeros(1)
path="StoredTrainingData\\PPO_A.pt"
path1="StoredTrainingData\\PPO_C.pt"

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    if os.path.exists(path):
        agent.Policy.actor=torch.load(path)

    #policy evaluation
    for t in range(max_number_of_stepsv):
        env.render()
        chosenAction,logpro=agent.policyAction(env.state)
        prevstate=env.state

        nextstate, reward, Terminal=env.step(chosenAction)
        agent.save_r_log(reward,logpro,chosenAction,prevstate)

        e_return+=reward
        if Terminal==True:
            break



    #if i%2==0:
        # policy improvement
    agent.Policy.train_net()

    if i % 29 == 0:
     print('score:',e_return,'\n',i ,'episode\n value:')
     agent.Policy.plot_cost()

    allrewards=np.append(allrewards,e_return)

torch.save(agent.Policy.actor, path)
torch.save(agent.Policy.critic, path1)
env.close()
x=np.asarray(range(len(allrewards)))
fig=plt.plot(x,allrewards)
plt.savefig("PPO.png")
plt.show()
sys.exit()
