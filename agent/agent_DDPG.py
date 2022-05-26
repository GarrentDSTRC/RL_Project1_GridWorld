import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

from agent.replay_buffer import ReplayBuffer
from agent.PrioritizedBuffer import NaivePrioritizedBuffer as Pbuffer
from agent.ou_noise import OUNoise

REPLAY_MEMORY = 100 # number of previous transitions to remember
BATCH_SIZE = 100 # size of minibatch
learning_rate=0.02
soft_tau=0.1

path="StoredTrainingData\\Qtab_Q.npy"

class PolicyNet(nn.Module):


    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(PolicyNet, self).__init__()
        self.data = []  # 存储轨迹
        self.L1=nn.Linear(in_features=n_states_num, out_features=hidden_size, bias=False)
        self.L2=nn.Linear(in_features=hidden_size, out_features=n_actions_num, bias=False)

        self.Qtab = np.load(path)

        for i in  self.parameters():
            i.data.fill_(0.1)

    def forward(self,inputs):
        x=F.relu(self.L1(inputs))
        x=self.L2(x)
        x=F.softmax(x,dim=1)

        return x
    def test_Q_getaction(self,state):
        state=np.asarray(state)
        action=[]
        if state.ndim!=0:
            for i in state:
                action.append(np.argmax(self.Qtab[:,int(i)]))
            return torch.tensor(action)
        else: return torch.tensor(np.argmax(self.Qtab[:,state]))

    def getactionDDPG(self,state):
        state = torch.as_tensor(state)
        if state.dim() != 0:
            states = torch.zeros(16,state.shape[0] )  # (*,16)
            for i in range(state.shape[0]):
                states[int(state[i]), i] = 1
        else:
            states = torch.zeros(16,1)
            states[int(state), 0] = 1
        # 将state转化成tensor one-hot vector 并且维度转化为[16]->[1,16]  unsqueeze(0)在第0个维度上切片

        prob = self.forward(states.t())  # 动作分布:

        #return torch.argmax(prob,1)
        # 从类别分布中采样1个动作
        m = torch.distributions.Categorical(prob)  # 生成分布
        action = m.sample()
        return action, prob
        #return action.item(), m.log_prob(action), prob


class CriticQ(nn.Module):

    def __init__(self, n_states_num, n_actions_num, hidden_size):
        super(CriticQ, self).__init__()
        self.data = []  # 存储轨迹
        self.L1=nn.Linear(in_features=n_states_num+n_actions_num, out_features=hidden_size, bias=False)
        self.L2=nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self,state,action):
        state = torch.as_tensor(state)
        action =torch.as_tensor(action)
        if state.dim() != 0:
            states = torch.zeros(16,state.shape[0] )  # (*,16)
            actions=torch.zeros(5,action.shape[0])
            for i in range(state.shape[0]):
                states[int(state[i]), i] = 1
                actions[int(action[i]),i]=1
        else:
            states = torch.zeros(16,1)
            states[int(state),0]=1
            actions = torch.zeros(5,1)
            actions[int(action),0]=1

        x=torch.cat([states.t(),actions.t()],1)
        x1=F.relu(self.L1(x))
        x2=self.L2(x1)

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
        self.actor = PolicyNet(n_states_num, n_actions_num, 128)
        self.actorT = PolicyNet(n_states_num, n_actions_num, 128)
        self.critic = CriticQ(n_states_num, n_actions_num, 128)
        self.criticT = CriticQ(n_states_num, n_actions_num, 128)

        self.exploration_noise = OUNoise(1)

        #优化器
        self.optimizerA = torch.optim.SGD(self.actor.parameters(), lr=learning_rate)
        self.optimizerC = torch.optim.SGD(self.critic.parameters(), lr=learning_rate)
        # replaybuffer
        self.buffer = Pbuffer(REPLAY_MEMORY )
        self.cbuffer = ReplayBuffer(REPLAY_MEMORY)

        self.cost_his = []

    def train_DDPG_common_buffer(self):
        minibatch = self.cbuffer.get_batch(BATCH_SIZE)
        a = list(zip(*minibatch))

        state = torch.FloatTensor(a[0])
        action = torch.LongTensor(a[1])
        reward = torch.FloatTensor(a[2])
        next_state = torch.FloatTensor(np.float32(a[3]))
        done = torch.FloatTensor(a[4])


        next_action_batch ,probbatchT= self.actorT.getactionDDPG(next_state)
        q_value_batch = self.criticT.forward(next_state, next_action_batch)
        #q_value_batch = self.critic.forward(next_state, next_action_batch)

        y_batch1=reward + (1.0-done)*self.gamma * q_value_batch

        self.optimizerC.zero_grad()
        Loss=((y_batch1-self.critic.forward(state,action))**2).mean()
        print("Q:",Loss)
        Loss.backward()
        #print("QN:", self.critic.L1.weight.grad)
        #print(self.critic.L1.weight.grad)
        self.optimizerC.step()


        """#update policy"""
        self.optimizerA.zero_grad()
        action,prob=self.actor.getactionDDPG(state)
        actionspace=[0,1,2,3,4]
        Qmax=[]
        for i in actionspace:
            Qmax.append(self.criticT.forward(state,i*torch.ones([state.shape[0]])))
        Qmax1=torch.cat(Qmax,1)
        #argmax 压缩
        Qmax2=torch.argmax(Qmax1,1)
        #one-hot编码
        Qmax3=torch.zeros([Qmax2.shape[0],5])
        for i in range(Qmax2.shape[0]):
            Qmax3[i,Qmax2[i]]=1

        q_gradient_batch=F.cross_entropy(prob,Qmax2)
        #q_gradient_batch = -self.critic.forward(state, self.actor.getactionDDPG(state))
        #s = torch.zeros(1,16)
        #s[0, 6] = 1
        #s1=torch.tensor([1.],requires_grad=True)

        #q_gradient_batch=((self.actor.getactionDDPG(s1)-0.5)**2)
        q_gradient_batch.mean().backward()
        print("P:",q_gradient_batch.mean())
        #print("PN:", self.actor.L1.weight.grad,self.actor.L2.weight.grad)
        self.optimizerA.step()


        # Update the target networks
        for target_param, param in zip(self.criticT.parameters(),self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.actorT.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )



    def train_net_common_buffer(self,vtab):
        minibatch=self.cbuffer.get_batch(BATCH_SIZE)
        a=list(zip(*minibatch))

        state = torch.FloatTensor(a[0])
        action = torch.LongTensor(a[1])
        reward = torch.FloatTensor(a[2])
        next_state = torch.FloatTensor(np.float32(a[3]))
        done = torch.FloatTensor(a[4])
        #weights = torch.FloatTensor(weights)


        policy_loss = []

        #losssum=torch.tensor(0.,requires_grad=True)
        Advantage=torch.zeros(BATCH_SIZE)

        for i in range(BATCH_SIZE):
            act,log_pro_act,pro=self.choose_action(state[i])

            if reward[i]==100:
                print(vtab)

            vtab[int(state[i])]+=learning_rate*(reward[i] + self.gamma * vtab[int(next_state[i])])

            Advantage[i] = float((reward[i] + self.gamma * vtab[int(next_state[i])]) - vtab[int(state[i])])

            loss = -torch.log(pro[0, action[i]]) *  float(Advantage[i])

            #loss.backward(retain_graph=True)
            policy_loss.append(loss.unsqueeze(0))

        self.optimizer.zero_grad()

        policy_loss1=torch.cat(policy_loss)

        #prios=torch.atan(policy_loss1)+1.58
        prios=torch.abs(Advantage)


        policy_loss2 = policy_loss1.sum()  # 求和


        # 反向传播
        #with torch.autograd.set_detect_anomaly(True):
        policy_loss2.backward(retain_graph=True)
        self.optimizer.step()
        self.cost_his.append(policy_loss2.item())

        return vtab

    def train_net(self,vtab):
        state, action, reward, next_state, done, indices, weights = self.buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        policy_loss = []

        #losssum=torch.tensor(0.,requires_grad=True)
        Advantage=torch.zeros(BATCH_SIZE)

        for i in range(BATCH_SIZE):
            act,log_pro_act,pro=self.choose_action(state[i])

            vtab[int(state[i])]+=learning_rate*(reward[i] + self.gamma * vtab[int(next_state[i])])

            Advantage[i] = float((reward[i] + self.gamma * vtab[int(next_state[i])]) - vtab[int(state[i])])

            loss = -torch.log(pro[0, action[i]]) *  float(Advantage[i]) * weights[i]

            #loss.backward(retain_graph=True)
            policy_loss.append(loss.unsqueeze(0))

        self.optimizer.zero_grad()

        policy_loss1=torch.cat(policy_loss)

        #prios=torch.atan(policy_loss1)+1.58
        prios=torch.abs(Advantage)

        self.buffer.update_priorities(indices, prios.data.cpu().numpy())

        policy_loss2 = policy_loss1.sum()  # 求和


        # 反向传播
        with torch.autograd.set_detect_anomaly(True):
         policy_loss2.backward(retain_graph=True)
        self.optimizer.step()
        self.cost_his.append(policy_loss2.item())

        return vtab


    #将状态传入神经网络 根据概率选择动作
    def  choose_action(self,state):
        state = torch.as_tensor(state)
        if state.dim()!=0:
            states=torch.zeros(state.size,16)  #(*,16)
            for i in range(state.size):
                states[state[i],i]=1
        else: states=torch.zeros(1,16)

        #将state转化成tensor one-hot vector 并且维度转化为[16]->[1,16]  unsqueeze(0)在第0个维度上切片
        #s = torch.Tensor(states).unsqueeze(0)
        prob = self.actor(states)  # 动作分布:
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



class agent_DDPG():

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
        #self.vtab[observe_state] = torch.sum(actionpro*(assume_reward + self.gamma * assume_nextv))

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


    def save_r_log(self,reward,logpro):
        self.Policy.put_data([reward,logpro])
