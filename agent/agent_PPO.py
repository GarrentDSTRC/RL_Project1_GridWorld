import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

#from replay_buffer import ReplayBuffer
from agent.PrioritizedBuffer import NaivePrioritizedBuffer as Pbuffer

REPLAY_MEMORY = 500 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
reward_decay=0.8
eps_clip=0.2
K_epochs=10
learning_rate=0.1

class actor(nn.Module):

    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(actor, self).__init__()
        self.data = []  # 存储轨迹
        self.actor_layer = nn.Sequential(
            nn.Linear(n_states_num,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions_num),
            nn.Softmax(dim=1)
        )

    def forward(self,inputs):
        return self.actor_layer(inputs)

class critic(nn.Module):

    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(critic, self).__init__()
        self.critic_layer = nn.Sequential(
            nn.Linear(n_states_num,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
        )

    def forward(self,inputs):
        return self.critic_layer(inputs)



class Policy():
    def __init__(self,n_states_num,n_actions_num):
        #状态数   state是一个16维向量，
        self.n_states_num = n_states_num
        #action是5维、离散，即上下左右
        self.n_actions_num = n_actions_num
        #学习率
        self.lr = learning_rate
        #gamma
        self.gamma = reward_decay
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        #网络
        self.actor = actor(n_states_num, n_actions_num, 128)
        self.critic=critic(n_states_num, n_actions_num, 128)

        self.actor_old = actor(n_states_num, n_actions_num, 128)
        self.critic_old=critic(n_states_num, n_actions_num, 128)

        self.critic_old.load_state_dict(self.critic.state_dict())
        self.actor_old.load_state_dict(self.actor.state_dict())
        #优化器
        self.optimizerA = torch.optim.SGD(self.actor.parameters(), lr=learning_rate)
        self.optimizerC = torch.optim.SGD(self.critic.parameters(), lr=learning_rate)

        self.MseLoss = nn.MSELoss()

        self.cost_his = []
        self.data = []

    def train_net(self):
        # 计算梯度并更新策略网络参数。tape为梯度记录器
        R = 0  # 终结状态的初始回报为0
        reward_to_go=[]
        for r, log_prob,choseA,prevstate in self.data[::-1]:  # 逆序取
            R = r + self.gamma * R  # 计算每个时间戳上的回报
            #rewards.append(R)
            reward_to_go.insert(0, R)
            #从0到T

        # Normalizing the rewards:
        reward_to_go = torch.tensor(reward_to_go)
        reward_to_go = (reward_to_go - reward_to_go.mean()) / (reward_to_go.std() + 1e-5)


        pack=list(zip(*self.data))

        old_rewards=torch.tensor(pack[0]).detach()
        old_states=torch.tensor(pack[3]).detach()
        old_actions=torch.tensor(pack[2]).detach()
        old_logpros=torch.tensor(pack[1]).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate_action(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logpros.detach())

            # Finding Surrogate Loss:
            #advantages = rewards - state_values.detach()
            advantages=torch.tensor(self.compute_gae(0,old_rewards,state_values,self.gamma)).detach()


            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actorloss = -torch.min(surr1, surr2)
            criticloss= + 0.5 * self.MseLoss(state_values, reward_to_go)
            loss=actorloss+criticloss- 0.01 * dist_entropy

            # take gradient step
            self.optimizerC.zero_grad()
            self.optimizerA.zero_grad()
            loss.mean().backward()
            self.optimizerC.step()
            self.optimizerA.step()

        self.cost_his.append(float(loss.mean()))

        # Copy new weights into old policy:
        self.critic_old.load_state_dict(self.critic.state_dict())


        self.data = []  # 清空轨迹

    def compute_gae(self,Last_next_value, rewards,  values, gamma=0.99, tau=0.95):
        values = torch.cat((values , torch.tensor(Last_next_value).unsqueeze(0)),0)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1]  - values[step]
            gae = delta + gamma * tau * gae
            returns.insert(0, gae)
        return returns

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
        prob = self.actor(states)  # 动作分布:
        # 从类别分布中采样1个动作
        m = torch.distributions.Categorical(prob)  # 生成分布
        action = m.sample()

        return action.item() , m.log_prob(action)

    #state action from batch
    def  evaluate_action(self,state,action):
        state = torch.as_tensor(state)
        if state.dim()!=0 :
            states=torch.zeros(state.shape[0],16)  #(*,16)
            for i in range(state.shape[0]):
                states[i,int(state[i])]=1
        else:
            states=torch.zeros(1,16)
            states[0,int(state)]=1

        #将state转化成tensor one-hot vector 并且维度转化为[16]->[1,16]  unsqueeze(0)在第0个维度上切片

        prob = self.actor(states)  # 动作分布:
        # 从类别分布中采样1个动作
        m = torch.distributions.Categorical(prob)  # 生成分布
        dist_entropy = m.entropy()

        action_logprobs =m.log_prob(action)
        action = m.sample()

        value=self.critic.forward(states)

        return action_logprobs, torch.squeeze(value),dist_entropy

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def put_data(self, item):
        # 记录r,log_P(a|s)z
        self.data.append(item)



class agent_PPO():

    #个性化设置
    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1,0]  # 上下左右

        self.size = 5

        self.gamma = reward_decay

        self.Policy=Policy(len(self.states),self.size)

        #模型知识

        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)


    def policyAction(self,observe_state):       #also do value_improvement
        self.state=observe_state

        chosenAction,logpro=self.Policy.choose_action(self.state)

        return chosenAction,logpro

    def save_r_log(self,reward,logpro,chosenAction,prevstate):
        self.Policy.put_data([reward,logpro,chosenAction,prevstate])
