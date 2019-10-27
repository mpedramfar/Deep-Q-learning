import random, torch
from collections import deque

class ReplayBuffer:
    def __init__(self, batch_size, buffer_size=int(1e5)):
        self.data = deque(maxlen=buffer_size)  
        self.batch_size = batch_size

    def store(self, e):
        self.data.append(e)

    def sample(self):
        size = min(self.batch_size, len(self.data))
        batch = random.sample(self.data, size)
        states, actions, rewards, next_states, dones = \
            [torch.tensor(x, dtype=torch.float32).view(size, -1) for x in zip(*batch)]
        return states, actions.long(), rewards, next_states, dones