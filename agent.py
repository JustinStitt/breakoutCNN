import numpy as np
import torch as T
from DQN import DeepQNetwork
from replay_memory import ReplayBuffer

class DDQNAgent():
    """
        A Double DQN agent has two networks. One local network and one target network.
        The local network is trained every iteration and is used for predictive action.
        The target network is updated to a soft copy of the local network every so often.

        The reason is because the Bellman equation would be valuing the network that is predicting
        as well as that same network being used to calculate loss. We have this separation of training
        and predicting to help the agent learn.
    """
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                    mem_size, batch_size, eps_min = 0.01, eps_dec = 5e-7,
                    replace = 10_000):
        pass
        self.gamma = gamma #used to discount future rewards
        self.epsilon = epsilon #used for epsilon-greedy action choosing algo.
        self.lr = lr #learning rate, essentially, how big of a step does the optimizer take
        self.n_actions = n_actions #number of actions available to our agent in its environment
        self.action_space = [i for i in range(n_actions)]#list comprehension to create array of indices of possible actions to choose from
        self.input_dims = input_dims #the dimensions of our input as defined by the agent's environment
        self.mem_size = mem_size #maximum amount of memories to store
        self.batch_size = batch_size #mini-batch size to sample from memory.
        self.eps_min = eps_min #smallest possible epsilon value for our agent
        self.eps_dec = eps_dec #how much to decrease epsilon each iteration
        self.replace_after = replace #how many iterations until we replace our target network with a sofy copy of our local network
        self.steps = 0 #iteration counter for use with replace_after

        #create a ReplayBuffer to store our memories, also used to sample a mini-batch
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.Q_local = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims = self.input_dims)
        self.Q_target = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims = self.input_dims)

    def store_memory(self, state, action, reward, state_, done):
        """
            Save a new memory to our ReplayBuffer
        """
        self.memory.store_memory(state, action, reward, state_, done)

    def sample_batch(self):
        """
            Pull a stochastic mini-batch from our ReplayBuffer
        """
        state, action, reward, state_, done = \
                            self.memory.sample_batch(self.batch_size)

        states = T.tensor(state).to(self.Q_local.device)
        actions = T.tensor(action).to(self.Q_local.device)
        rewards = T.tensor(reward).to(self.Q_local.device)
        states_ = T.tensor(state_).to(self.Q_local.device)
        dones = T.tensor(done).to(self.Q_local.device)

        return states, actions, rewards, states_, dones


    def choose_action(self, observation):
        """
            Choose an action from our action space using an epsilon-greedy algorithm.
            We can either EXPLOIT, or EXPLORE based on a random probability.

            Exploiting will choose the best known action. (confidence)

            Exploring will explore a random action. This will possibly present new information to our agent
            to learn from.
        """
        if np.random.random() > self.epsilon:#epsilon-greedy (EXPLOIT)
            state = T.tensor([observation], dtype = T.float).to(self.Q_local.device)
            actions = self.Q_local.forward(state)
            action = T.argmax(actions).item()#.item() gets index from tensor
        else:#(EXPLORE)
            action = np.random.choice(self.action_space)#choose random action from our action space

        return action

    def replace_target_network(self):
        """
            after replace_after iterations we update our target network
            to be a soft copy of our local network
        """
        if self.replace_after is not None and \
                    self.steps % self.replace_after == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())

    def decrement_epsilon(self):
        """
            decrease epsilon, but not below eps_min
        """
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def learn(self):
        """
            Main part of our agent.

            First we zero the gradient of our optimzier to stop exploding gradients.
            Then we sample a stochastic mini-batch from our ReplayBuffer.

            Then we make predictions and evaluations of this random mini-batch, step our optimzer
            and calculate loss.

            Finally, we decrement our epsilon and begin the cycle of (SEE->DO->LEARN) once again.
        """
        if self.memory.mem_cntr < self.batch_size:#if we dont have a full batch of memories, dont learn quite yet
            return

        self.Q_local.optimizer.zero_grad()#zero out our gradient for optimzer. Stop exploding gradients

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_batch()

        indices = np.arange(self.batch_size)

        q_pred = self.Q_local.forward(states)[indices, actions]#local pred
        q_next = self.Q_target.forward(states_)#target pred
        q_eval = self.Q_local.forward(states_)

        max_actions = T.argmax(q_eval, dim = 1)
        q_next[dones] = 0.0#set to not done

        q_target = rewards + self.gamma*q_next[indices, max_actions]#bellman equation
        loss = self.Q_local.loss(q_target, q_pred).to(self.Q_local.device)
        loss.backward()#back-propagation

        self.Q_local.optimizer.step()
        self.steps += 1

        self.decrement_epsilon()

    def save_agent(self):
        self.Q_local.save_model('local')
        self.Q_target.save_model('target')

    def load_agent(self):
        self.Q_local.load_model('local')
        self.Q_target.load_model('target')
