import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


cuda_enabled = T.cuda.is_available()

save_path = 'saved_models/saved_model_'

class DQN_CNN(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DQN_CNN,self).__init__()

        #convolutional layers
        #first convolutional layer takes 4 in_channels. (last 4 frames of the environment)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 8, stride = 4)#in_channels = 1 because luma (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        #fully connected linear layers
        conv_out = self.calculate_conv_out_dims(input_dims)
        self.fc4 = nn.Linear(conv_out, 512)#find smarter way to find in_channels instead of 7 * 7 * 64
        self.fc5 = nn.Linear(512, n_actions)#output layer

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()

        if cuda_enabled:
            T.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            T.set_default_tensor_type('torch.Float')
        self.device = T.device('cuda:0' if (cuda_enabled) else 'cpu')
        self.to(self.device)

    def calculate_conv_out_dims(self, input_dims):
        empty_state = T.zeros( 1,1,input_dims[0],input_dims[1] )
        x = self.conv1(empty_state)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, state):
        #convolutional pass
        state = state.view(state.shape[0],state.shape[3],state.shape[1],state.shape[2])
        #print('-==state shape==-', state.shape)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #linear pass and reshaping with view
        x = F.relu(self.fc4(x.view(x.size()[0], -1)))
        return self.fc5(x)#no activation for output layer. Maybe consider softmax?

    def save_model(self, type):
        print('...saving {} model...'.format(type))
        path = save_path + type + '.pt'
        T.save(self.state_dict(), path)

    def load_model(self, type):
        print('...loading {} model...'.format(type))
        path = save_path + type + '.pt'
        self.load_state_dict(T.load(path))


#note: Gym has Discrete or Box inputs from the environment. CNN's should be used with Box ??? observation types only ???

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 25000,
                eps_min = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.replace_cntr = 0
        self.replace_target_after = 250

        self.Q_local = DQN_CNN(lr,  input_dims = input_dims, n_actions = n_actions)
        self.Q_target = DQN_CNN(lr,  input_dims = input_dims, n_actions = n_actions)

        sz = self.mem_size
        dims = input_dims

        self.state_memory = np.zeros((sz,*dims), dtype = np.float32)
        self.new_state_memory = np.zeros((sz,*dims), dtype = np.float32)
        self.action_memory = np.zeros(sz, dtype = np.int64)
        self.reward_memory = np.zeros(sz, dtype = np.float32)
        self.terminal_memory = np.zeros(sz, dtype = np.bool)#done flags

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon: #epsilon-greedy algo
            state = T.tensor([observation], dtype=T.float).to(self.Q_local.device)
            actions = self.Q_local.forward(state)
            action = T.argmax(actions).item()#returns index
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_local.optimizer.zero_grad()
        self.replace_target_network()
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace = False)

        batch_index = np.arange(self.batch_size)#[0,1,2,3,4,5,self.batch_size - 1]

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_local.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_local.device)
        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_local.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_local.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_local.device)

        q_pred = self.Q_local.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_eval = self.Q_local.forward(new_state_batch)

        max_actions = T.argmax(q_eval,dim = 1)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*q_next[batch_index, max_actions]
        loss = self.Q_local.loss(q_target, q_pred).to(self.Q_local.device)
        loss.backward()

        self.Q_local.optimizer.step()
        self.replace_cntr += 1
        self.decrement_epsilon()


    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def replace_target_network(self):
        if self.replace_cntr is not None and self.replace_cntr % self.replace_target_after == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())

    def save_agent(self):
        #save local and target network
        self.Q_local.save_model("local")
        self.Q_target.save_model("target")

    def load_agent(self):
        #load local and target network
        self.Q_local.load_model("local")
        self.Q_target.load_model("target")
