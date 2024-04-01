import random

class QLearning:
    
    def __init__(self, alpha, gamma, epsilon, actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}
        self.loss = 0
        
    def convert(self, state):
        state = ""
        for i in range(len(state)):
            state += str(state[i])
        return state
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def init_state_in_q_table(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
    
    def choose_action(self, state):
        state = self.convert(state)
        self.init_state_in_q_table(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda action: self.get_q_value(state, action))
        
    def learn(self, state, action, reward, next_state):
        state = self.convert(state)
        next_state = self.convert(next_state)

        self.init_state_in_q_table(next_state)
        
        q_value = self.get_q_value(state, action)
        
        best_q_value = max([self.get_q_value(next_state, next_action) for next_action in self.actions])
        
        self.loss += (reward + self.gamma * best_q_value - q_value) ** 2
        
        self.q_table[(state, action)] = q_value + self.alpha * (reward + self.gamma * best_q_value - q_value)
