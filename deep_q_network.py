import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # convolutions to process observations and pass then to fully connected layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)  # input_dims[0]: number of channels, 32: number of 
        # outgoing filters, 8: kernel size (8*8 pixels)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # 32: number of incoming filters, 64: number of outgoing filters
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # convolutions to process observations and pass then to fully connected layers

        processed_input_dims = self.calculate_output_dims(input_dims)
        
        self.fc1 = nn.Linear(processed_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # use GPU if available
        self.to(self.device)  # move whole model to device

    def calculate_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))  # np.prod: to return the product of array elements over a given axis.

    def forward(self, state):  # forward propagation includes defining layers
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))  # conv3 shape is batch size * number of filters * H * W (of output image)

        conv_state = conv3.view(conv3.size()[0], -1) # means that get the first dim and flatten others

        flat1 = F.relu(self.fc1(conv_state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

