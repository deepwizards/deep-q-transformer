# replay_buffers.py

from collections import deque, namedtuple
import random
import numpy as np

# A named tuple representing a single transition in your environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """A simple FIFO experience replay buffer for DQN agents."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """A replay buffer with prioritized sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, *args, priority=1.0):
        """Saves a transition with a given priority."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6):
        """
        Samples a batch of transitions, with probabilities proportional to priority**alpha.
        Returns the sampled batch and their corresponding indices.
        """
        scaled_priorities = np.array(self.priorities) ** alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(range(len(self.memory)), size=batch_size, p=sample_probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, new_priorities):
        """Updates the priorities of sampled transitions."""
        for idx, new_priority in zip(indices, new_priorities):
            self.priorities[idx] = new_priority

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
