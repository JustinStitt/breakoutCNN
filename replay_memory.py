import numpy as np

class ReplayBuffer(object):
    """
        The ReplayBuffer class will store memories for our agent.
        A "replay" or "memory" consists of 5 channels.
        1) State - the state given to the agent from the environment
        2) Action - the action that the agent chose to take
        3) Reward - the discounted future reward that is expected from this action
        4) State_ - the resulting state
        5) Done - done flag to tell if the epoch is ended on this state_
            (agent dies, or crashes into a wall, etc...)
    """
    def __init__(self, max_size,  input_shape,n_actions):
        """
            First we declare a maximum size. This is how many memories we can store.
            Once we store the maximum amount of memories that we can. We have wrap-around indexing
            that replaces our oldest memory with the newest one.

            Next, our memory counter will keep track of which memory we are currently on. We have initialized
            the capacity to max_size, but index we start inserting at is mem_cntr. Also used for wrap-around indexing.

            Lastly, for each of our 5 channels, we initialize a numpy array of zeros with the correct dimensions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        #N-d memory channels
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype = np.float32)
        #1-d memory channels
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype = np.bool)

    def store_memory(self, state, action, reward, state_, done):
        """
            Storing a memory requires 5 channels.
            State - current observation
            action - action our agent took
            reward - the discounted future reward our agent perceived from its action
            state_ - the new state resulting from the environment progressing one step
            done - done flag to tell if our episode is done (agent died, timed-out, etc...)
        """
        index = self.mem_cntr % self.mem_size#wrap-around indexing
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.mem_cntr += 1 #we have stored a memory, so now increment memory counter

    def sample_batch(self, batch_size):
        """
            Get a batch of size 'batch_size' from our saved memories.
            'batch' is not a single index but rather a range
            so when we, for example, do:  states = self.state_memory[batch],
            we are using numpy array slicing to slice a range of states not just one.
        """
        max_mem = min(self.mem_cntr, self.mem_size)#we want the active size of our memories not the capacity
        #mem_cntr has wrap-around indexing so it is important to make sure we take the smaller of the index and the capacity
        batch = np.random.choice(max_mem, batch_size, replace = False)#stochastic ranged-based indicies used for slicing our numpy arrays below

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, states_, dones
