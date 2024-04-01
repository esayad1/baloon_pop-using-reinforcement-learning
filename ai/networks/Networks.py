import random
import torch
import torch.nn as nn
import torch.nn.init as init

############################################################################################################

class DQNNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_size)
        )
        
        # Initialisation des poids de manière aléatoire
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

############################################################################################################

class PNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(PNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Initialisation des poids de manière aléatoire
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.seq(x)

############################################################################################################

class VNetwork(nn.Module):
    def __init__(self, input_size):
        super(VNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  
        )
        
        # Initialisation des poids de manière aléatoire
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return self.seq(x)

