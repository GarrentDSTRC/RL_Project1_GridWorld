import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

EPSILON=0.4
ep_d=0.005
Sigma=0.05

#pip install pyglet -i https://pypi.tuna.tsinghua.edu.cn/simple

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

class PolicyGradient():
    def __init__(self,n_states_num,n_actions_num,learning_rate=0.01,reward_decay=0.8 ):
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
        # 存储轨迹  存储方式为  （每一次的reward，动作的概率）
        self.data = []
        self.cost_his = []
    #存储轨迹数据
    def put_data(self, item):
        # 记录r,log_P(a|s)z
        self.data.append(item)
    def train_net(self):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        R = 0  # 终结状态的初始回报为0
        policy_loss = []
        for r, log_prob in self.data[::-1]:  # 逆序取
            R = r + self.gamma * R  # 计算每个时间戳上的回报
            # 每个时间戳都计算一次梯度
            loss = -log_prob * R
            policy_loss.append(loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 求和
        #反向传播
        policy_loss.backward()
        self.optimizer.step()
        self.cost_his.append(policy_loss.item())
        self.data = []  # 清空轨迹
    #将状态传入神经网络 根据概率选择动作


    def choose_action_e_greedy(self, state):
        # 将state转化成tensor 并且维度转化为[16]->[1,16]  unsqueeze(0)在第0个维度上田间
        s = torch.Tensor(state).unsqueeze(0)
        prob = self.pi(s)  # 动作分布:

        global EPSILON

        if random.random() < EPSILON:
            sigma = Sigma
            prob1=torch.tensor(prob)
            for i in range(prob.shape[1]):
                prob1[0,i]=prob[0,i]+ random.gauss(0, sigma)
            prob2 = F.softmax(prob1,dim=1)
            m = torch.distributions.Categorical(prob2)
        # 从类别分布中采样1个动作
        else:
            m = torch.distributions.Categorical(prob)  # 生成分布
        action = m.sample()

        EPSILON -= ep_d

        return action.item(), m.log_prob(action)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



class agent_PG():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1,0]  # 上下左右

        self.size = 5

        self.gamma = 0.8  # 折扣因子

        self.PG=PolicyGradient(len(self.states),self.size)


    def policyAction(self,observe_state):
        self.state=observe_state

        state=torch.zeros(16)
        state[self.state]=1
        chosenAction,logpro=self.PG.choose_action_e_greedy(state)

        return chosenAction, logpro
    def save_r_log(self,reward,logpro):
        self.PG.put_data([reward,logpro])
