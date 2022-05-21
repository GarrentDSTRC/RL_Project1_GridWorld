import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

#from replay_buffer import ReplayBuffer
from agent.PrioritizedBuffer import NaivePrioritizedBuffer as Pbuffer

REPLAY_MEMORY = 500 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch

class PolicyNet(nn.Module):


    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(PolicyNet, self).__init__()
        self.data = []  # 存储轨迹
        self.L1=nn.Linear(in_features=n_states_num, out_features=hidden_size, bias=False)
        self.L2=nn.Linear(in_features=hidden_size, out_features=n_actions_num, bias=False)

    def forward(self,inputs):
        x=F.relu(self.L1(inputs))
        x=self.L2(x)
        x=F.softmax(x,dim=1)
        return x

class Policy():
    def __init__(self,n_states_num,n_actions_num,learning_rate=0.01,reward_decay=0.8):
        #状态数   state是一个16维向量，
        self.n_states_num = n_states_num
        #action是5维、离散，即上下左右
        self.n_actions_num = n_actions_num
        #学习率
        self.lr = learning_rate
        #gamma
        self.gamma = reward_decay
        #网络
        self.pi = PolicyNet(n_states_num, n_actions_num, 128)
        #优化器
        self.optimizer = torch.optim.SGD(self.pi.parameters(), lr=learning_rate)

        self.cost_his = []
        self.data = []
    def train_net(self,vtab,qtab):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        R = 0  # 终结状态的初始回报为0
        policy_loss = []
        for r, log_prob,choseA,s in self.data[::-1]:  # 逆序取
            #R = r + self.gamma * R  # 计算每个时间戳上的回报

            # 每个时间戳都计算一次梯度
            Advantage=qtab[int(s),int(choseA)]-vtab[int(s)]
            loss = -log_prob * int(Advantage)
            policy_loss.append(loss)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 求和
        # 反向传播
        policy_loss.backward()
        self.optimizer.step()
        self.cost_his.append(policy_loss.item())
        self.data = []  # 清空轨迹

    #将状态传入神经网络 根据概率选择动作
    def  choose_action(self,state):
        state = torch.as_tensor(state)
        if state.dim()!=0:
            states=torch.zeros(state.size,16)  #(*,16)
            for i in range(state.size):
                states[state[i],i]=1
        else:
            states=torch.zeros(1,16)
            states[0,int(state)]=1

        #将state转化成tensor one-hot vector 并且维度转化为[16]->[1,16]  unsqueeze(0)在第0个维度上切片
        #s = torch.Tensor(states).unsqueeze(0)
        prob = self.pi(states)  # 动作分布:
        # 从类别分布中采样1个动作
        m = torch.distributions.Categorical(prob)  # 生成分布
        action = m.sample()
        return action.item() , m.log_prob(action), prob

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def put_data(self, item):
        # 记录r,log_P(a|s)z
        self.data.append(item)



class agent_DP_p():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1,0]  # 上下左右

        self.size = 5

        self.gamma = 0.8  # 折扣因子

        self.Policy=Policy(len(self.states),self.size)

        #模型知识
        self.rewards = np.ones([self.size, len(self.states)], dtype=int) * -1  # 回报的数据结构为np
        self.rewards[1, 0] = 100
        self.rewards[2, 5] = 100

        self.terminate_states = np.zeros(len(self.states))  # 终止状态为np格式
        self.terminate_states[4] = 1

        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)

        self.vtab = torch.zeros([16], dtype=float,requires_grad=False)
        self.qtab = torch.zeros([16,5], dtype=float, requires_grad=False)

        self.gamma = 0.8  # 折扣因子



    def policyAction(self,observe_state):       #also do value_improvement
        self.state=observe_state
        actionSpace = [0, 1, 2, 3,4]

        chosenAction,logpro,actionpro=self.Policy.choose_action(self.state)

        # value_improvement under  policy with stochastic action
        assume_nextv=torch.zeros(self.size)
        assume_reward = torch.zeros(self.size)
        for i in actionSpace:
            as_n, Re,Ter=self.step(i)
            assume_nextv[i] = self.vtab[as_n]
            assume_reward[i] = Re

        # use stomastic action
        self.vtab[observe_state] = torch.sum(actionpro*(assume_reward + self.gamma * assume_nextv))
        self.qtab[observe_state,int(chosenAction)]=assume_reward[int(chosenAction)]+self.gamma*assume_nextv[int(chosenAction)]

        return chosenAction, logpro

    def step(self,action):             #action 0123
        # 系统当前状态
        state = self.state
        as_n = self.state + self.actions[action]

        if self.terminate_states[state]:
            return state, 100, True

        elif  ((as_n not in self.states))or\
                (as_n ==6 or as_n== 8 or as_n== 9) or(action==2 and  self.state % self.size == 0) or (action == 3 and self.state % self.size == 3):

            return state, -2, False
        else:
            return as_n, -1,False


    def save_r_log(self,reward,logpro,chosenAction,prevstate):
        self.Policy.put_data([reward,logpro,chosenAction,prevstate])
