import random
import torch
import torch.nn as nn

from collections import deque
import numpy as np

from ai.networks.Networks import DQNNetwork

class Memory:
    
    def __init__(self, buffer_size: float = 1000, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        
    def add_transition(self, transition: tuple, priority: float) -> None:
        self.buffer.append(transition)
        self.priorities.append(priority)
        
    def sample_batch(self, batch_size: int) -> tuple:
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(np.arange(len(self.buffer)), size=batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights
    
    def update_priorities(self, indices: list, priorities: list) -> None:
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority
            
    def __len__(self) -> int:
        return len(self.buffer)

class DDQNWPER:
    """
    Double Deep Q Network with Prioritized Experience Replay
    """    
    
    def __init__(self, lr, gamma, epsilon, input_size, actions):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.network = DQNNetwork(input_size, len(actions))
        self.target_network = DQNNetwork(input_size, len(actions))
        
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        self.loss = 0
        
        self.bach_size = 32
        self.memory = Memory()
        
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
          
    def add_in_memory(self, state, action, reward, next_state) -> None:
        self.memory.add_transition((state, action, reward, next_state), abs(reward) + 1e-5)
       
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def learn(self):
        if len(self.memory.buffer) < self.bach_size:
            return
        
        batch, indices, weights = self.memory.sample_batch(self.bach_size)
        
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        q_values = self.network(states)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1)[0]
            next_q_values = rewards + self.gamma * next_q_values
            
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
        weights = torch.tensor(weights, dtype=torch.float32)
    
        loss = (weights * (q_values - next_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        
        self.memory.update_priorities(indices, (q_values - next_q_values).abs().detach().numpy().tolist())
        
        self.loss = loss.item()
        