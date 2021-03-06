import numpy as np
import random


class agent_Q():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1]  # 上下左右

        self.size = 4


        #self.vtab = np.zeros([16, ], dtype=float)
        self.Qtab=np.zeros([self.size,len(self.states)],dtype=float)

        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)
        self.gamma = 0.8  # 折扣因子




    def policyAction(self,observe_state,epsilon):
        actionSpace = [0, 1, 2, 3]
        self.state=observe_state

        for i in range(self.size):
            assume_nextstate = self.state + self.actions[i]  #boundery check
            if (assume_nextstate not in self.states) or(i==2 and  self.state % self.size == 0) or (i == 3 and self.state % self.size == 3):
                actionSpace.remove(i)

        Vmax=-900

        if random.random()<epsilon:
            return random.choice(actionSpace)
        else:
            for i in actionSpace:
                #assume_nextstate = self.state + self.actions[i]
                #assume_reward=self.getReward(i)
                Vmax2=self.Qtab[i,self.state]
                if Vmax2>Vmax:
                    Vmax=Vmax2
                    chosenAction=i

            return chosenAction