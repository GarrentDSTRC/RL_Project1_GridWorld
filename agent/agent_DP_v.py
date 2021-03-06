import numpy as np
import random


class agent_DP_V():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1,0]  # 上下左右不动

        self.size = 4

        self.rewards = np.ones([self.size, len(self.states)], dtype=int) * -1;  # 回报的数据结构为np
        self.rewards[1, 0] = 100
        self.rewards[2, 5] = 100

        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)

        self.vtab = np.zeros([16, ], dtype=float)

        self.gamma = 0.8  # 折扣因子
        self.count=-1

    def getReward(self,action):  #这里给01234
        return self.rewards[action,self.state]

    def policyAction(self,observe_state,epsilon):
        actionSpace = [0, 1, 2, 3,4]
        self.state=observe_state

        for i in range(self.size):
            assume_nextstate = self.state + self.actions[i]  #boundery check
            if (assume_nextstate not in self.states) or(i==2 and  self.state % self.size == 0) or (i == 3 and self.state % self.size == 3):
                actionSpace.remove(i)
        Vmax=-100

        if random.random()<epsilon:
            return random.choice(actionSpace)
        else:
            for i in actionSpace:
                assume_nextstate = self.state + self.actions[i]
                assume_reward=self.getReward(i)
                Vmax2=assume_reward+self.gamma*self.vtab[assume_nextstate]
                if Vmax2>Vmax:
                    Vmax=Vmax2
                    chosenAction=i

            return chosenAction


    def deterministicPolicy(self, observe_state, epsilon):

        actionSpace = [0, 1, 2, 3]
        self.state = observe_state
        action1=[0,0,0,2,2,2,1,0]
        action2=[0,0,0,2,2,1,2,0]
        self.count+=1

        if self.count==16:
            self.count=0
        elif self.count>=8:
            return action1[self.count-8]
        return action2[self.count]


