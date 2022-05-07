import numpy as np
import random


class agent_DP_V():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1 ,0]  # 上下左右

        self.size = 5

        self.rewards = np.ones([self.size, len(self.states)], dtype=int) * -1;  # 回报的数据结构为np
        self.rewards[1, 0] = 100
        self.rewards[2, 5] = 100

        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)

        self.vtab = np.zeros([16, ], dtype=float)

        self.gamma = 0.8  # 折扣因子
        self.count=-1

    def getReward(self,action):  #这里给0123
        return self.rewards[action,self.state]

    def policyAction(self,observe_state,epsilon):
        actionSpace = [0, 1, 2, 3, 4]
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


    def deterministicPolicy(self, observe_state,policy):

        d={"上":0 ,"下":1 ,"左":2 ,"右":3 ,"不动":4}

        self.state = observe_state
        s_a_tab_p2=np.asarray([d['下'],d['左'],d['左'],d['左'],
                               d['不动'],d['上'],-1, d['上'],
                               -1,-1,d['右'],d['上'],
                               d['右'],d['右'],d['右'],d['上'],
                                ])
        s_a_tab_p1 = np.asarray([d['右'],d['下'],d['左'],d['左'],
                                 d['不动'],d['左'],-1, d['上'],
                                 -1, -1, d['右'], d['上'],
                                 d['右'], d['右'],d['上'], d['左'],
                                 ])
        if policy==1:
            return s_a_tab_p1[self.state]
        else: return s_a_tab_p2[self.state]


