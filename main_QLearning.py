import sys
import time

import gym
import environment.grid_env as grid
import numpy as np
from matplotlib import pyplot as plt
import agent.agent_Q as ag
import os


env=grid.GridEnv1()
agent=ag.agent_Q()
num_episodes = 200
max_number_of_steps=100
epsilon=0.8
dicrease=0.01
a=0.2   #LearningRate More cautious than model based
allrewards=np.zeros(1)
path="StoredTrainingData\\Qtab_Q.npy"

for i in range(num_episodes):
    env.reset()
    e_return=0
    epsilon -= dicrease
    if os.path.exists(path):
        agent.Qtab=np.load(path)
    for t in range(max_number_of_steps):
        env.render()
        chosenAction=agent.policyAction(env.state,epsilon)
        nextstate, reward, Terminal=env.step(chosenAction)

        agent.Qtab[chosenAction, env.state] = agent.Qtab[chosenAction, env.state] + \
                                              a * (reward + agent.gamma * np.max(agent.Qtab[:, nextstate]) - agent.Qtab[
            chosenAction, env.state])  # Dynamic Programming -Value iterations

        env.state=nextstate
        e_return+=reward
        if Terminal==True:
            break

        time.sleep(0.01)

    if i%20==0:
     print('score:',e_return,'\n',i ,'episode\n value:',agent.Qtab)

    allrewards=np.append(allrewards,e_return)


np.save(path,agent.Qtab)
env.close()
x=np.asarray(range(len(allrewards)))
fig=plt.plot(x,allrewards)
plt.savefig("Qlearning.png")
plt.show()
sys.exit()
