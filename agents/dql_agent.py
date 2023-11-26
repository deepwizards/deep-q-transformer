import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.efficient_transformer import EfficientTransformerQNetwork
from models.state_embedding import StateEmbedding
from agents.replay_buffers import PrioritizedReplayBuffer
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DoubleQLearningAgentUCB:
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_heads, lr, gamma, buffer_size, batch_size, embedding_dim):
        self.state_embedding = StateEmbedding(input_dim, embedding_dim)
        self.current_net = EfficientTransformerQNetwork(embedding_dim, hidden_dim, output_dim, n_layers, n_heads)
        self.target_net = EfficientTransformerQNetwork(embedding_dim, hidden_dim, output_dim, n_layers, n_heads)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(list(self.current_net.parameters()) + list(self.state_embedding.parameters()), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.criterion = torch.nn.SmoothL1Loss()
        self.ucb_c = 2
        self.action_counts = np.zeros(output_dim)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        state_action_values = self.current_net(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values[done_batch] = reward_batch[done_batch]

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        errors = torch.abs(state_action_values - expected_state_action_values).tolist()
        self.memory.update_priority(range(self.batch_size), errors)

    def select_action(self, state, global_step):
        state_embedded = self.state_embedding(state)
        with torch.no_grad():
            action_values = self.current_net(state_embedded)
        ucb_values = action_values + self.ucb_c * torch.sqrt(torch.log(global_step) / (self.action_counts + 1e-5))
        action = ucb_values.max(1)[1].view(1, 1)
        self.action_counts[action.item()] += 1
        return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
