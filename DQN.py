import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

cuda_enabled = T.cuda.is_available()
print('Cuda Enabled: ', cuda_enabled)

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(DeepQNetwork, self).__init__()

        self.save_dir = 'trained_model/'
        #model architecture
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        conv_out_dims = self.calculate_conv_out_dims(input_dims)
        #we need to calculate the dimensions of what is outputted after our convolutional pass
        self.fc1 = nn.Linear(conv_out_dims, 512)#first fully-connected layer
        self.fc2 = nn.Linear(512, n_actions)#output layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)#define optimizer. Adam (adaption of RMS)
        self.loss = nn.MSELoss()#define loss function (mean-squared error)
        #set device to cuda on GPU if cuda is enabled
        if cuda_enabled:#set default tensor data type if we have cuda capabilities
            T.set_default_tensor_type('torch.cuda.FloatTensor')#use gpu
        self.device = T.device('cuda:0' if (cuda_enabled) else 'cpu')
        self.to(self.device)

    def calculate_conv_out_dims(self, input_dims):
        """
            After our convolution layers are done we need to know the
            input dimensions so that we can pass it to our fully-connected layers
            This is an aux function to calculate the input dims by just passing a tensor of zeros
            sequentially and taking the shape afterwards for use in our first fc layer.
        """
        state = T.zeros(1, *input_dims)
        dim = self.conv1(state)
        dim = self.conv2(dim)
        dim = self.conv3(dim)
        #dimensions are equal the the product of all out_channels from conv. pass + integer cast to ensure integer dimensionality
        return int(np.prod(dim.size()))

    def forward(self, state):
        """
            Pass a state sequentially through our network.
            First through the covolutional layers then through the fully-connected layer(s)
            to ultimately generate an array of actions which we can argmax() later to find
            the action to actually choose.
        """
        #pass our intial state through our convolutional layers sequentially
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], - 1)#tensor.view recasts the shape (similar to unsqueeze(0))
        #now pass our output after the last convolutional pass to our first fc layer and eventually our output layer
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        return actions

    def save_model(self, type):
        print('...saving {} model...'.format(type))
        path = self.save_dir + type +'.pt'
        T.save(self.state_dict(), path)

    def load_model(self, type):
        print('...loading {} model...'.format(type))
        path = self.save_dir + type + '.pt'
        self.load_state_dict(T.load(path))

#
