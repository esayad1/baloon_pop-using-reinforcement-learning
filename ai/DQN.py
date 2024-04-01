import random
import torch
import torch.nn as nn

from ai.networks.Networks import DQNNetwork

class DQN:
    
    def __init__(self, lr, gamma, epsilon, input_size, actions):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.network = DQNNetwork(input_size, len(actions))
        self.loss = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        
    def convert(self, state) -> None:
        state = [int(i) for i in state]
        return torch.tensor(state, dtype=torch.float32)
    
    def choose_action(self, state) -> int:
        state = self.convert(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            with torch.no_grad():
                return self.network(state).argmax().item()
            
    def learn(self, state, action, reward, next_state):
        state = self.convert(state)
        next_state = self.convert(next_state)
        q_values = self.network(state)
        next_q_values = self.network(next_state)
        q_values[action] = reward + self.gamma * next_q_values.max().item()
        loss = self.criterion(self.network(state), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss = loss.item()    
