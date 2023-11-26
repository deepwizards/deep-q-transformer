import torch
import torch.nn as nn
from collections import deque
import random

class DeepQLearningModel(nn.Module):
    """
    A Deep Q-Learning model.

    Args:
        input_dim (int): Dimension of the input.
        action_dim (int): Dimension of the action space.
    """
    def __init__(self, input_dim, action_dim):
        super(DeepQLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        """
        Forward pass through the Deep Q-Learning Model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor representing action values.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ExperienceReplay:
    """
    Experience Replay Buffer for Deep Q-Learning.

    Args:
        capacity (int): Maximum number of experiences to store in the buffer.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        """
        Push a new experience to the buffer.

        Args:
            state (Tensor): The state observed from the environment.
            action (int): The action taken.
            reward (float): The reward received.
        """
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Size of the batch to sample.

        Returns:
            list: A list of sampled experiences.
        """
        random.seed(42)  # Optional: for reproducibility
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Get the current size of internal memory.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.buffer)

if __name__ == "__main__":
    # Example initialization and testing of the models can be added here
    pass
