import sys
import time

import gym
import environment.grid_env_nn as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_DDPG as ag
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


env=grid.GridEnv1_nn()
agent=ag.agent_DDPG()
num_episodes = 300
max_number_of_stepsv=100
max_number_of_stepsp=100
epsilon=0.8
dicrease=0.01
a=0.2   #LearningRate
allrewards=np.zeros(1)
path1="StoredTrainingData\\DDPGT.pt"
path2="StoredTrainingData\\DDPG.pt"
if os.path.exists(path1):
    agent.Policy.criticT = torch.load(path1)
    agent.Policy.critic = torch.load(path2)

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    #
    for t in range(max_number_of_stepsv):
        env.render()
        chosenAction,prob=agent.Policy.actorT.getactionDDPG(env.state)
        prevstate=env.state


        nextstate, reward, Terminal=env.step(chosenAction)

        agent.Policy.cbuffer.add(prevstate,chosenAction, reward, nextstate,Terminal)
        #agent.Policy.buffer.push(prevstate,chosenAction, reward, nextstate,Terminal)

        e_return+=reward
        if Terminal==True:
            break



    if i%5==0 and agent.Policy.cbuffer.count()==agent.Policy.cbuffer.size():
    #if i % 5 == 0 and agent.Policy.buffer.__len__() == agent.Policy.buffer.capacity:

        # train and erase to ensure is experience is on-policy
        #agent.vtab=agent.Policy.train_DDPG_common_buffer(agent.vtab)
        #agent.Policy.cbuffer.erase()
        agent.Policy.train_DDPG_common_buffer()


    if i % 100 == 0:
        print('score:',e_return,'\n',i ,'episode\n value:')
        for j in range(5):
            for i in range(16):
                print(float(agent.Policy.criticT.forward(i,j)),float(agent.Policy.critic.forward(i,j)))
            print('\n')
        #for target_param, param in zip(agent.Policy.criticT.parameters(),agent.Policy.critic.parameters()):
            #print(target_param,param)
        #print(agent.vtab)
        #agent.Policy.plot_cost()



    allrewards=np.append(allrewards,e_return)

torch.save(agent.Policy.criticT,path1)
torch.save(agent.Policy.critic,path2)

env.close()
x=np.asarray(range(len(allrewards)))
fig=plt.plot(x,allrewards)
plt.savefig("DDPG.png")
plt.show()
sys.exit()
