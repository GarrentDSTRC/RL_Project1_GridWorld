import sys
import time

import gym
import environment.grid_env_nn as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_PG as ag
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


env=grid.GridEnv1_nn()
agent=ag.agent_PG()
num_episodes = 200
max_number_of_steps=100
epsilon=0.8
dicrease=0.01
a=0.2   #LearningRate
allrewards=np.zeros(1)
path="StoredTrainingData\\PG.pt"

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    if os.path.exists(path):
        agent.PG.pi=torch.load(path)
    for t in range(max_number_of_steps):
        env.render()
        chosenAction,logpro=agent.policyAction(env.state)
        nextstate, reward, Terminal=env.step(chosenAction)
        agent.save_r_log(reward,logpro)


        env.state=nextstate
        e_return+=reward
        if Terminal==True:
            break

    agent.PG.train_net()
        #time.sleep(0.01)

    if i%40==0:
     print('score:',e_return,'\n',i ,'episode\n value:')
     agent.PG.plot_cost()

    allrewards=np.append(allrewards,e_return)

torch.save(agent.PG.pi,path)

env.close()
x=np.asarray(range(len(allrewards)))
fig=plt.plot(x,allrewards)
plt.savefig("PG.png")
plt.show()
sys.exit()
