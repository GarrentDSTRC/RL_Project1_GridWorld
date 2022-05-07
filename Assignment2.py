import numpy as np
import agent.agent_DP_v as ag
import environment.grid_env as grid
import time

gamma=0.8
env=grid.GridEnv1()

agent=ag.agent_DP_V()
P_pi_1=np.zeros([16,16],dtype=int)
P_pi_2=np.array(P_pi_1)
r_pi_1=np.zeros(16)
r_pi_2=np.array(r_pi_1)

#Question 1

for i in agent.states:

    chosenAction=agent.deterministicPolicy(i,1)
    env.state=i
    nextstate, reward, Terminal = env.step(chosenAction)
    P_pi_1[i,nextstate]=1
    r_pi_1[i]=agent.rewards[chosenAction,i]
 #VALUE=0 means the forbiden state, VLUE=100 means the target state. others are -1

    chosenAction = agent.deterministicPolicy(i,2)
    nextstate, reward, Terminal = env.step(chosenAction)
    P_pi_2[i,nextstate]=1
    r_pi_2[i] =agent.rewards[chosenAction,i]
    print("For policy 1: r_pi=\n",r_pi_1,"\nP_pi=\n",P_pi_1,"\n For policy 2: r_pi=\n",r_pi_2,"\nP_pi:\n",P_pi_2)

#Question 2
v_pi_1=np.dot(np.linalg.inv(np.identity(P_pi_1.shape[0])-gamma*P_pi_1),r_pi_1.reshape(len(r_pi_1),1))
v_pi_2=np.dot(np.linalg.inv(np.identity(P_pi_2.shape[0])-gamma*P_pi_2),r_pi_2.reshape(len(r_pi_2),1))
Ptrans=np.array(list(range(1,17)))
print("v_pi_1:\n",v_pi_1,"\nv_pi_2\n",v_pi_2,"\n状态转移策略1\n",np.dot(P_pi_1,Ptrans.reshape(len(Ptrans),1)),"\n状态转移策略2\n",np.dot(P_pi_2,Ptrans.reshape(len(Ptrans),1)))

#print(Ptrans.reshape(len(Ptrans),1),"=",r_pi_1.reshape(len(r_pi_1),1),"+",gamma,np.dot(P_pi_1,Ptrans.reshape(len(Ptrans),1)))
#Question 3
num_episodes = 100
max_number_of_steps=20
dicrease=0.01
allrewards=np.zeros(1)

for p in [1, 2]:
    agent.vtab = np.zeros(len(agent.vtab))
    for i in range(num_episodes):
        env.reset()  #random choose the init state , or the agent will not go through some state
        e_return=0

        for t in range(max_number_of_steps):
            env.render()

            #chosenAction=agent.policyAction(env.state,epsilon)

            chosenAction=agent.deterministicPolicy(env.state,p)
            nextstate, reward, Terminal=env.step(chosenAction)

            agent.vtab[env.state]=reward+agent.gamma*agent.vtab[nextstate]  #Dynamic Programming -Value iterations

            env.state=nextstate
            #if Terminal==True:
                #break

            #time.sleep(0.05)

    print('Iteration_solution_Policy ',p,':\n',agent.vtab.reshape(len(agent.vtab),1))
