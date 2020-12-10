import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelDQNet:
    name = 'DuelDQNet'
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.V = nn.Linear(self.fc2_dims, 1)
        self.A = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A

    def save_checkpoint(self):
        print('Saving Checkpoint...')
        T.save(self.state_dict(), './checkpoints/DuelDQNet')
    
    def load_checkpoint(self):
        print('Loading Checkpoint...')
        self.load_state_dict(T.load('./checkpoints/DuelDQNet'))