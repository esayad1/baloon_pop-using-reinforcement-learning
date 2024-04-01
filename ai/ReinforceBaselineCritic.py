import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ai.networks.Networks import PNetwork, VNetwork

class ReinforceBaselineCritic:
    
    def __init__(self, lr, gamma, epsilon, input_size, actions):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        
        self.policy = PNetwork(input_size, len(actions)) 
        self.critic = VNetwork(input_size)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.probs = []
        self.rewards = []
        self.states = []
        
        self.loss = 0
        
    def convert(self, state) -> None:
        state = [int(i) for i in state]
        return torch.tensor(state, dtype=torch.float32)
    
    def choose_action(self, state) -> int:
        state = self.convert(state)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.probs.append(m.log_prob(action))
        self.states.append(state)
        return action.item()
    
    def learn(self) -> None:
        discounts = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounts.insert(0, R)
            
        discounts = torch.tensor(discounts, dtype=torch.float32)
        
        advantages = discounts - self.critic(torch.stack(self.states)).squeeze()
        
        policy_loss = []
        
        for proba, advantage in zip(self.probs, advantages):
            policy_loss.append(-proba * advantage)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        self.loss = policy_loss.item()
        
        self.optimizer.zero_grad()
        self.optimizer_critic.zero_grad()
        
        loss_final = policy_loss + nn.MSELoss()(self.critic(torch.stack(self.states)).squeeze(), discounts)
    
        loss_final.backward()
        
        self.optimizer.step()
        
        self.probs = []
        self.rewards = []
        self.states = []
        
            
    
            
    