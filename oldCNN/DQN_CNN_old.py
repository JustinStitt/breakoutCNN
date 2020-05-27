#in DeepQ Network set up we want 3 classes
#1) DQN class
#2) Agent class
#3) ReplayBuffer
#followed tutorial from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/deep_q_network.py

#step 1... imports
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#step 2... DQN class inheriting from torch.nn module
#https://pytorch.org/docs/stable/nn.html
T.set_default_tensor_type('torch.cuda.FloatTensor')#use gpu
class DeepQNetwork(nn.Module):

    """
        our DeepQNetwork class is our ML/RL model
    """
    def __init__(self, lr, n_actions, input_dims):
        super(DeepQNetwork,self).__init__()#invoke super constructor of pytorch.nn module
        self.conv1 = nn.Conv2d(input_dims[0], 32, 1, stride=4)#convolutional layer # 1
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        #stride-- https://deepai.org/machine-learning-glossary-and-terms/stride
        #in short, more stride is more movement across input tensor
        self.conv2 = nn.Conv2d(32, 64, 1, stride=2)#the input
        self.conv3 = nn.Conv2d(64,64,1,stride=1)
        #for each successive conv layer, the in_channels must be equal to previous conv layer out_channels
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        #we need to calculate the dimensions of our inputs post-convolution. Then feed that to fc1
        self.fc1 = nn.Linear(fc_input_dims, 512)#fully-connected layer 1
        self.fc2 = nn.Linear(512, n_actions)#fully connected layer 2
        #choose our optimizer. Adam is good, lets use RMSprop here though for testing
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        #choose our loss function (Mean Squared Error)
        self.loss = nn.MSELoss()
        #use GPU as it is much better at handling tensors, if no GPU. use def. cpu
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)#cast to chosen device

    def calculate_conv_output_dims(self,input_dims):
        """
            Given input_dims we need to find out what our convolutional
            process yields as far as output dims, we can then use this
            for our fully-conected layer #1 input dims.
        """
        state = T.zeros(1,*input_dims)#create tensor (1,unpacked_input_dims
        #state = state.view((1,1,4,4))
        #state = T.zeros(1,4,8,8)
        #state = T.randn(20,4,50,100)
        #successively pass a 0-tensor through our convolutional layers and find the output dimensions
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        """
            Given a state, find Q-values for every action in our action space.
            'state' can be seen as input
        """
        #relu activation function see: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0],-1)
        #conv2d.view reshapes our tensor see: https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        #now pass our conv_state (post convolutional steps) to our "flat" or fully-conected neural network layers
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), 'checkpoints/cnn_model.pt')
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load('checkpoints/cnn_model.pt'))

class ReplayBuffer():
    """
        stores Agent memories for actions, rewards, terminal (Done-flags) and state + new_state
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape),
                                    dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype = np.float32)

        #memory of all our actions in episode. dtype = int64 because large size potential
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        #memory of the rewards the Agent got for each action. dtype = float32 because rewards are floats
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        #memory of done flags, dtype = bool because we only need T/F
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Agent():
    """
        What an agent for any DNN needs.
        our DQNAgent will inherit from this stock Agent.
    """
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                mem_size, batch_size, eps_min = 0.01, eps_dec = 5e-7,
                replace = 1000):
        self.gamma = gamma #discount factor for future rewards
        self.epsilon = epsilon #used for epsilon-greedy algo.
        self.lr = lr #learning rate
        self.n_actions = n_actions # num. of actions in our action space
        self.input_dims = input_dims #dimensions of input tensor
        self.batch_size = batch_size #how many steps of our optimizer per episode
        self.action_space = [i for i in range(n_actions)]#[0,1,2,3,4, ... , len(n_actions) - 1]
        self.learn_step_counter = 0
        self.eps_min = eps_min #smallest possible epsilon for our agent
        self.eps_dec = eps_dec #decrement amount for epsilon
        self.replace_target_cnt = replace

        self.memory = ReplayBuffer(mem_size,input_dims, n_actions)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state,action,reward,state_, done)

    def choose_action(self, observation):
        raise NotImplementedError

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    def decrement_epsilon(self):
        #print('new epsilon: ',self.epsilon)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                            self.eps_min else self.eps_min
    def sample_memory(self):
        state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    def learn(self):
        raise NotImplementedError



class DQNAgent(Agent):
    """
        DQNAgent inherits from Agent
    """
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims = self.input_dims)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims = self.input_dims)
    def choose_action(self, observation):
        """
            Epsilon-Greedy Algorithm
        """
        if np.random.random() > self.epsilon:#exploit
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:#explore
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        """
            The bread and butter of our DQN.
            Runs optimizer, checks batch sizes, back-propagates.
            calculates Q-vals.
        """
        if self.memory.mem_cntr < self.batch_size:#dont learn if we dont have a complete batch
            return
        self.q_eval.optimizer.zero_grad()#helps against exploding gradients

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]

        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
