import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import ReplayBuffer

def LinearModel(in_size, out_size, h = [64, 64]):
    h = [in_size] + h + [out_size]
    layers = []
    for i in range(len(h) - 1):
        layers += [nn.Linear(h[i], h[i+1]), nn.ReLU()]
    return nn.Sequential(*layers[0:-1])
    

class DQNAgent:
    def __init__(self, env, state_size, action_size, 
                 batch_size, gamma, lr, update_every, tau,
                 eps_start, eps_end, eps_decay, seed):
        
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)
        
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.Q_target = LinearModel(state_size, action_size)
        self.Q_local = LinearModel(state_size, action_size)
        
        self.memory = ReplayBuffer(batch_size=batch_size)
        self.optim = torch.optim.Adam(self.Q_local.parameters(), lr=lr)
        
        self.update_counter = 0

    def env_reset(self, train_mode=True):
        return self.env.reset()
    
    def env_step(self, action):
        return self.env.step(action)
    
    def env_render(self, train_mode=False):
        return self.env.render()

    def env_close(self, train_mode=True):
        if not train_mode:
            return self.env.close()
    
    def get_action(self, state, epsilon=0.):
        if random.random() < epsilon:
            return np.random.choice(np.arange(self.action_size))        
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.Q_local.eval()
        with torch.no_grad():
            action = np.argmax(self.Q_local(state).data.numpy())
        return action
    
    def step(self, state, action, reward, next_state, done):
        self.memory.store( (state, action, reward, next_state, 1 if done else 0) )
        
        self.update_counter = (self.update_counter+1) % self.update_every
        if self.update_counter == 0:
            self.update_Q()

    def update_Q(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        Q_target_next = self.Q_target(next_states).detach().max(dim=1, keepdim=True)[0]
        Q_target_pred = rewards + self.gamma * Q_target_next * (1.0 - dones)
        self.Q_local.eval()
        Q = self.Q_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q, Q_target_pred)
        self.Q_local.train()
        self.Q_local.zero_grad()
        loss.backward()
        self.optim.step()
        
        for t_param, l_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            t_param.data.copy_(self.tau*l_param.data + (1.0-self.tau)*t_param.data)
            
    def train(self, num_episodes, max_t=1000, is_finished=None, render=False):
        scores = []
        eps = self.eps_start
        
        for i in range(num_episodes):
            state = self.env_reset(train_mode=True)
            score = 0
            for _ in range(max_t):
                action = self.get_action(state, eps)
                if render: self.env_render(train_mode=True)
                next_state, reward, done, _ = self.env_step(action)
                self.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done: break
        
            eps = max(self.eps_end, eps*self.eps_decay)
            scores.append(score)
            if is_finished and is_finished(scores, num_episodes):
                break
        if render: self.env_close(train_mode=False)
        return scores

    def run(self, num_episodes=1, max_t=1000, render=None):
        if render == None: render = num_episodes==1
        scores = []
        for i in range(num_episodes):
            state = self.env_reset(train_mode=False)
            score = 0
            for _ in range(max_t):
                action = self.get_action(state)
                if render: self.env_render(train_mode=False)
                next_state, reward, done, _ = self.env_step(action)
                score += reward
                state = next_state
                if done: break
    
            scores.append(score)
            if render: self.env_close(train_mode=False)
        return scores
