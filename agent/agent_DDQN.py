import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

from agent.replay_buffer import ReplayBuffer
from agent.PrioritizedBuffer import NaivePrioritizedBuffer as Pbuffer
from agent.ou_noise import OUNoise

REPLAY_MEMORY = 3000 # number of previous transitions to remember
BATCH_SIZE = 200 # size of minibatch
learning_rate=0.02
soft_tau=0.1
epsilon=0.9
epsilon_d=0.0001


path="StoredTrainingData\\Qtab_Q.npy"


class CriticQ(nn.Module):

    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(CriticQ, self).__init__()
        self.data = []  # 存储轨迹
        self.L1=nn.Linear(in_features=n_states_num, out_features=hidden_size, bias=False)
        self.L2=nn.Linear(in_features=hidden_size, out_features=n_actions_num, bias=False)
        self.Lin = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
    def forward(self,state):
        state = torch.as_tensor(state)
        if state.dim() != 0:
            states = torch.zeros(state.shape[0], 16)  # (*,16)
            for i in range(state.shape[0]):
                states[i,int(state[i]) ] = 1
        else:
            states = torch.zeros(1,16)
            states[0,int(state)]=1

        x1=F.relu(self.L1(states))
        xin=F.relu(self.Lin(x1))
        x2=self.L2(xin)

        return x2

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

        self.critic = CriticQ(n_states_num, n_actions_num, 40)
        self.criticT = CriticQ(n_states_num, n_actions_num, 40)

        #self.t = CriticQ(n_states_num, n_actions_num, 40)
        #self.tT = CriticQ(n_states_num, n_actions_num, 40)
        #self.optimizert = torch.optim.SGD(self.t.parameters(), lr=learning_rate)

        self.exploration_noise = OUNoise(1)

        #优化器

        self.optimizerC = torch.optim.SGD(self.critic.parameters(), lr=learning_rate)
        # replaybuffer
        self.buffer = Pbuffer(REPLAY_MEMORY )
        self.cbuffer = ReplayBuffer(REPLAY_MEMORY)

        self.cost_his = []
        self.Qtab = np.load(path)

    def train_DDQN_common_buffer(self):
        minibatch = self.cbuffer.get_batch(BATCH_SIZE)
        a = list(zip(*minibatch))

        state = torch.FloatTensor(a[0])
        action = torch.LongTensor(a[1])
        reward = torch.FloatTensor(a[2])
        next_state = torch.FloatTensor(np.float32(a[3]))
        done = torch.FloatTensor(a[4])

        actionspace = [0, 1, 2, 3, 4]

        self.critic.forward(next_state)

        #获取Q下标argmax
        Qmax=[]
        for i in actionspace:
            Qmax.append(self.critic.forward(next_state,i*torch.ones([next_state.shape[0]])))
        Qmax1=torch.cat(Qmax,1)
        #argmax 压缩
        Qmax2=torch.argmax(Qmax1,1)
        #one-hot编码
        Qmax3=torch.zeros([Qmax2.shape[0],5])
        for i in range(Qmax2.shape[0]):
            Qmax3[i,Qmax2[i]]=1



        q_value_batch = self.critic.forward(next_state, Qmax2)
        #q_value_batch = self.critic.forward(next_state, next_action_batch)

        y_batch1=reward + (1.0-done)*self.gamma * q_value_batch

        #update Q
        self.optimizerC.zero_grad()
        Loss=((y_batch1-self.critic.forward(state,action))**2).mean()
        print("Q:",Loss)
        Loss.backward()
        #print("QN:", self.critic.L1.weight.grad)
        #print(self.critic.L1.weight.grad)
        self.optimizerC.step()
        self.cost_his.append(float(Loss))


        # Update the target networks
        for target_param, param in zip(self.criticT.parameters(),self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def train_DDQN_P_buffer(self):
        state, action, reward, next_state, done, indices, weights = self.buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        #print(state,action,reward,next_state)

        actionspace = [0, 1, 2, 3, 4]
        #获取Q下标argmax
        Qmax1=self.critic.forward(next_state)
        Qmax2 = torch.argmax(Qmax1, 1)

        qst_1 = self.criticT.forward(next_state).gather(1,Qmax2.unsqueeze(1)).squeeze(1)
        #qst_1 = self.criticT.forward(next_state).max(1)[0]
        #q_value_batch = self.critic.forward(next_state, next_action_batch)

        y_batch1=(reward + (1.0-done)*self.gamma * qst_1 ).detach()

        Qst=self.critic.forward(state).gather(1,action.unsqueeze(1))
        #update Q
        self.optimizerC.zero_grad()
        #Loss=(((y_batch1-self.critic.forward(state,action).squeeze(1))*weights)**2)
        Loss = (( y_batch1 - Qst.squeeze(1) )  ** 2)*weights

        prios= Loss + 1e-5
        self.buffer.update_priorities(indices, prios.data.cpu().numpy())

        Loss1=Loss.mean()
        print("QLOSS:",Loss1)
        Loss1.backward()
        #print("QPara:", self.critic.L1.weight.grad,self.criticT.L1.weight.grad)
        #print(self.critic.L1.weight.grad)
        self.optimizerC.step()
        self.cost_his.append(float(Loss1))


    def compute_td_loss(self):
        state, action, reward, next_state, done, indices, weights = self.buffer.sample(BATCH_SIZE)

        state = (torch.FloatTensor(np.float32(state)))
        next_state = (torch.FloatTensor(np.float32(next_state)))
        action = (torch.LongTensor(action))
        reward = (torch.FloatTensor(reward))
        done = (torch.FloatTensor(done))
        weights = (torch.FloatTensor(weights))

        q_values = self.critic(state)
        next_q_values = self.criticT(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizerC.zero_grad()
        loss.backward()
        self.buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizerC.step()

        return loss

    def copyTargetnetwork(self):

        # Update the target networks
        for target_param, param in zip(self.criticT.parameters(),self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                #param
            )
    #将状态传入神经网络 根据概率选择动作
    def getactionQ(self, state, epsilon):
        actionSpace = [0, 1, 2, 3,4]
        Q=[]
        if random.random() < epsilon:
            return random.choice(actionSpace)
        else:
            return  torch.argmax(self.criticT.forward(state))

    def test_Q_getaction(self,state, epsilon):
        state=np.asarray(state)
        action=[]
        if state.ndim!=0:
            for i in state:
                action.append(np.argmax(self.Qtab[:,int(i)]))
            return torch.tensor(action)
        else: return torch.tensor(np.argmax(self.Qtab[:,state]))

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



class agent_DDQN():

    def __init__(self):
        self.states = list(range(16))  # 状态空间 0-15
        self.actions = [-4, 4, -1, 1,0]  # 上下左右

        self.size = 5

        self.Policy=Policy(len(self.states),self.size)


        self.states.remove(6)
        self.states.remove(9)
        self.states.remove(8)



        self.epsilon=epsilon
        self.epsilon_d=epsilon_d


    def policyAction(self,observe_state):       #also do value_improvement
        self.state=observe_state

        chosenAction=self.Policy.getactionQ(self.state,self.epsilon)
        #chosenAction=self.Policy.test_Q_getaction(self.state,self.epsilon)
        self.epsilon-=self.epsilon_d

        return chosenAction


    def save_r_log(self,reward,logpro):
        self.Policy.put_data([reward,logpro])
